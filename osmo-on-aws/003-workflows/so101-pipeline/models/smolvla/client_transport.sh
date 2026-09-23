#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Name the transport the shared robot client should use. SOURCED — it exports TRANSPORT_MODULE.
#
# Nothing to install: this model's transport is the `transport.py` slot, injected beside the
# client, and its only dependencies (pyzmq, msgpack, msgpack-numpy) are already in the robot
# task's base install because every model here speaks msgpack over ZMQ.
#
# The module is named after the SLOT, not after the model. The slot has a fixed name, so the
# pipeline never learns a per-model module name — which is what lets one robot task serve both.
set -euo pipefail

export TRANSPORT_MODULE="transport"
echo "[smolvla] transport: ${TRANSPORT_MODULE}"
