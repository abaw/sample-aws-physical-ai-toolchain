#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Provision the transport the shared robot client should use to reach THIS model's server, and
# name it. SOURCED, not executed — it exports TRANSPORT_MODULE.
#
# This exists because the transport is the model's choice, not the robot's: contract B fixes the
# observation and action shapes and says nothing about the bytes. Without this slot the robot
# task would install one model's package to drive every model, which is exactly the coupling the
# composed pipeline exists to remove.
#
# --no-deps: only the PolicyClient is wanted, not the training stack. GR00T's own pins for numpy
# and torch are deliberately not honoured — they predate lerobot's, which this arm's calibrated
# joint space depends on.
set -euo pipefail

GROOT_COMMIT="${GROOT_COMMIT:-1a1837f20538b7d7e21f977a11a5aee14f99803c}"

pip install --quiet --root-user-action=ignore --no-deps \
  "gr00t @ https://github.com/NVIDIA/Isaac-GR00T/archive/${GROOT_COMMIT}.tar.gz"

# transformers is needed only because gr00t imports Gr00tPolicy at package level.
pip install --quiet --root-user-action=ignore \
  "transformers==${TRANSFORMERS_VERSION:-5.16.1}"

export TRANSPORT_MODULE="gr00t.policy.server_client"
echo "[gr00t-n17] transport: ${TRANSPORT_MODULE}"
