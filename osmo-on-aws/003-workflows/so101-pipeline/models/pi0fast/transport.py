#!/usr/bin/env python3
"""ZMQ REQ/REP transport shared by the pi0-FAST policy server and the robot client.

The robot client is `so101-vla/files/eval_so101_bounded.py`, reused unchanged: every line of
its hardware safety — envelope clamping, per-step delta limits, temperature confirmation, the
release path, torque read-back — was earned on this arm and must not exist in two copies that
can drift apart. It takes `--transport-module`, and this is what that points at.

Nothing here is model-specific. It is a request/reply socket carrying msgpack-numpy, and it
exists separately only because the client's default transport ships inside another model's
package and pulling that in to serve this one would be absurd.

**`allow_pickle=False` is a hard boundary.** msgpack_numpy will happily wire an arbitrary
object through `numpy.frombuffer` with pickle, which on a server that accepts network input is
remote code execution. So msgpack is driven directly with an encoder that refuses anything it
does not recognise, rather than trusting the library's own default hook.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
import zmq

# Set on both ends. Small enough to be a real bound, large enough for two uncompressed
# 640x480 RGB frames (~1.8 MB) plus overhead.
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


class MsgSerializer:
    """msgpack with numpy support and no pickle path, in either direction."""

    _ND = b"__nd__"

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(
            data, object_hook=MsgSerializer._decode, raw=False, strict_map_key=False
        )

    @staticmethod
    def _encode(obj: Any) -> dict:
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                # The exact case that would otherwise reach pickle.
                raise TypeError(
                    "refusing to serialize an object-dtype array: convert it to a concrete "
                    "dtype first, because the fallback path for this is pickle"
                )
            return {
                MsgSerializer._ND: True,
                b"shape": list(obj.shape),
                b"dtype": obj.dtype.str,
                b"data": obj.tobytes(order="C"),
            }
        if isinstance(obj, np.generic):
            return {b"__scalar__": True, b"dtype": obj.dtype.str, b"data": obj.tobytes()}
        raise TypeError(f"refusing to serialize {type(obj).__name__}; no pickle fallback")

    @staticmethod
    def _decode(obj: dict) -> Any:
        # `is True` rather than truthiness: msgpack's int and bool type codes are distinct,
        # and a 1 must not be accepted where a marker is expected.
        if obj.get(MsgSerializer._ND) is True:
            array = np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"]))
            return array.reshape(tuple(obj[b"shape"]))
        if obj.get(b"__scalar__") is True:
            return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"]))[0]
        return obj


class PolicyClient:
    """Request/reply client. Constructor and `get_action` match what the robot client calls.

    One socket per instance and a receive timeout set at construction, because the client
    switches between a generous set-up timeout and a short control-loop one by rebuilding
    rather than mutating. A timed-out REQ socket cannot be reused — ZMQ's REQ state machine
    refuses a second send before a reply — so a timeout rebuilds it here, which is what lets
    the caller retry while the arm holds its last commanded pose.
    """

    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.context = zmq.Context.instance()
        self.socket = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: Any) -> Any:
        payload = MsgSerializer.to_bytes({"endpoint": endpoint, "data": data})
        try:
            self.socket.send(payload)
            reply = MsgSerializer.from_bytes(self.socket.recv())
        except zmq.error.Again:
            # Rebuild before propagating, so the caller's retry has a usable socket.
            self._connect()
            raise
        if isinstance(reply, dict) and reply.get("error"):
            raise RuntimeError(f"Server error: {reply['error']}")
        return reply

    def ping(self) -> bool:
        """True if the server answered. **Never raises**, which is the contract.

        The robot client polls this bare — `if self.client.ping():` — while the server is
        still fetching a checkpoint and loading a model, so a timeout here is the normal
        case and must read as "not ready yet". Raising instead ends the run on the first
        poll, before the server has had a chance to come up.
        """
        try:
            return bool(self._call("ping", None))
        except zmq.error.ZMQError:
            # `_call` has already rebuilt the socket: a timed-out REQ cannot send again.
            return False

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        reply = self._call("get_action", {"observation": observation, "options": None})
        return reply["action"], reply.get("info") or {}

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None


class PolicyServer:
    """Reply server. `handler(observation) -> action dict` does the model-specific work."""

    def __init__(self, handler, host: str = "0.0.0.0", port: int = 5555):  # noqa: S104
        self.handler = handler
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
        self.socket.bind(f"tcp://{host}:{port}")
        self.host, self.port = host, port

    def serve_forever(self) -> None:
        while True:
            raw = self.socket.recv()
            try:
                request = MsgSerializer.from_bytes(raw)
                endpoint = request.get("endpoint")
                if endpoint == "ping":
                    reply: Any = True
                elif endpoint == "get_action":
                    action, info = self.handler(request["data"]["observation"])
                    reply = {"action": action, "info": info}
                else:
                    reply = {"error": f"unknown endpoint {endpoint!r}"}
            except Exception as exc:  # noqa: BLE001
                # Replied rather than raised. A REP socket that does not answer wedges the
                # client's REQ state machine, so the caller would see a timeout and never
                # learn what actually went wrong.
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            self.socket.send(MsgSerializer.to_bytes(reply))

    def close(self) -> None:
        self.socket.close(linger=0)
