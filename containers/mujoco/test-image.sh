#!/bin/bash
# =============================================================================
# CPU smoke test of a built MuJoCo training image
#
# Runs containers/mujoco/test_train_entrypoint.py inside the image, so the test
# gets the image's mujoco/playground/jax, GL libraries and ffmpeg. No GPU is
# needed: JAX runs on the CPU and MuJoCo renders through OSMesa.
#
# Usage: containers/mujoco/test-image.sh [image]
#   image - the image to test (default: mujoco-training:latest)
# =============================================================================

set -euo pipefail

IMAGE="${1:-mujoco-training:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The test and the entrypoint it runs come from this checkout, mounted read-only;
# pytest is installed for the run only, so the pushed image stays as built.
# The import check comes first: the test skips (and pytest passes) when
# mujoco_playground does not import, which here would hide a broken image.
docker run --rm \
    -v "$HERE:/workspace/containers/mujoco:ro" \
    -e MUJOCO_GL=osmesa \
    -e PYOPENGL_PLATFORM=osmesa \
    "$IMAGE" \
    sh -c "python -c 'import mujoco, mujoco.mjx, mujoco_playground, jax' && pip install --no-cache-dir --quiet pytest==8.4.2 && python -m pytest -p no:cacheprovider -rs containers/mujoco/test_train_entrypoint.py"
