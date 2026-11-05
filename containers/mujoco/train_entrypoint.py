"""
MuJoCo Playground RL Training Entrypoint

Trains a policy with Brax PPO on a MuJoCo Playground (MJX) environment, writes
Brax checkpoints and a rollout MP4 of the trained policy to OUTPUT_DIR, and
uploads OUTPUT_DIR to S3 when CHECKPOINT_BUCKET is set.

Environment variables (set by the Batch job definition or container overrides):
  ENV_NAME          - Playground env name (default: CartpoleBalance)
  NUM_ENVS          - Parallel environments (default: 2048)
  NUM_TIMESTEPS     - Total environment steps (default: 5000000)
  SEED              - Random seed (default: 0)
  CHECKPOINT_BUCKET - S3 bucket for checkpoints and the MP4 (default: none)
  CHECKPOINT_PREFIX - S3 key prefix (default: checkpoints/<job_id>)
  OUTPUT_DIR        - Local output dir (default: /tmp/checkpoints/<job_id>)

Smoke run on a CPU (no GPU, no bucket):
  JAX_PLATFORMS=cpu NUM_ENVS=8 NUM_TIMESTEPS=8000 python train_entrypoint.py
"""

import functools
import os
import shutil
from pathlib import Path

import jax
import mediapy
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)

# Rollout video frame size
VIDEO_HEIGHT = 240
VIDEO_WIDTH = 320


def ppo_config(env_name):
    """Returns Playground's tuned Brax PPO config for env_name."""
    if env_name in registry.dm_control_suite.ALL_ENVS:
        return dm_control_suite_params.brax_ppo_config(env_name)
    if env_name in registry.locomotion.ALL_ENVS:
        return locomotion_params.brax_ppo_config(env_name)
    if env_name in registry.manipulation.ALL_ENVS:
        return manipulation_params.brax_ppo_config(env_name)
    raise ValueError(f"Unknown Playground env: {env_name}. Available envs: {registry.ALL_ENVS}")


def scale_to_run(params, num_envs, num_timesteps):
    """Shrinks the tuned config so a small run (a CPU smoke run) stays small.

    The tuned batch is sized for thousands of GPU envs. With fewer envs than
    the batch size, Brax would unroll each env many times per training step,
    so the batch shrinks to the env count. Brax runs at least
    num_evals * num_resets_per_eval training steps, so both are capped by the
    number of training steps the timestep budget pays for.
    """
    if num_envs < params.batch_size:
        params.batch_size = num_envs
    steps_per_update = (
        params.batch_size
        * params.num_minibatches
        * params.get("unroll_length", 10)
        * params.get("action_repeat", 1)
    )
    updates = max(1, num_timesteps // steps_per_update)
    params.num_evals = min(params.num_evals, updates)
    if "num_resets_per_eval" in params:
        params.num_resets_per_eval = max(1, min(params.num_resets_per_eval, updates // params.num_evals))
    params.num_eval_envs = min(params.get("num_eval_envs", 128), num_envs)
    # Brax rolls out num_envs envs until it has batch_size * num_minibatches
    # trajectories, so the env count has to divide that evenly
    trajectories = params.batch_size * params.num_minibatches
    if trajectories % num_envs != 0:
        raise ValueError(
            f"NUM_ENVS={num_envs} must divide batch_size * num_minibatches = {trajectories}"
            f" ({params.batch_size} * {params.num_minibatches}); try a power of two"
        )


def write_rollout(env, make_inference_fn, params, episode_length, seed, path):
    """Plays the trained policy for one episode and writes it as an MP4."""
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    state = reset_fn(rng)
    trajectory = [state]
    for _ in range(episode_length):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)
        trajectory.append(state)
        if state.done:
            break

    frames = env.render(trajectory, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    # The image installs ffmpeg; elsewhere fall back to the imageio-ffmpeg binary
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(str(path), frames, fps=1.0 / env.dt)
    print(f"  Rollout: {len(frames)} frames -> {path}")


def upload_dir(local_dir, bucket, prefix):
    """Uploads every file under local_dir to s3://bucket/prefix/."""
    import boto3

    s3 = boto3.client("s3")
    uploaded = 0
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            key = f"{prefix}/{f.relative_to(local_dir)}"
            print(f"  Uploading {f} -> s3://{bucket}/{key}")
            s3.upload_file(str(f), bucket, key)
            uploaded += 1
    print(f"Done. Uploaded {uploaded} files to s3://{bucket}/{prefix}/")


def main():
    # =========================================================================
    # 1. READ PARAMETERS FROM ENV
    # =========================================================================
    job_id = os.environ.get("AWS_BATCH_JOB_ID", "local")
    env_name = os.environ.get("ENV_NAME", "CartpoleBalance")
    num_envs = int(os.environ.get("NUM_ENVS", "2048"))
    num_timesteps = int(os.environ.get("NUM_TIMESTEPS", "5000000"))
    seed = int(os.environ.get("SEED", "0"))
    bucket = os.environ.get("CHECKPOINT_BUCKET", "")
    prefix = (os.environ.get("CHECKPOINT_PREFIX") or f"checkpoints/{job_id}").strip("/")
    output_dir = Path(os.environ.get("OUTPUT_DIR", f"/tmp/checkpoints/{job_id}")).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MuJoCo Playground RL Training (Brax PPO)")
    print(f"  Env:         {env_name}")
    print(f"  Num envs:    {num_envs}")
    print(f"  Timesteps:   {num_timesteps}")
    print(f"  Seed:        {seed}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # =========================================================================
    # 2. TRAIN
    # =========================================================================
    env = registry.load(env_name)
    params = ppo_config(env_name)
    params.num_envs = num_envs
    params.num_timesteps = num_timesteps
    scale_to_run(params, num_envs, num_timesteps)

    train_params = params.to_dict()
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in train_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **train_params.pop("network_factory")
        )

    def progress(step, metrics):
        reward = metrics.get("eval/episode_reward")
        print(f"  step {step}: eval reward {reward}")

    # Brax writes one orbax checkpoint per eval under checkpoint_dir/<step>
    make_inference_fn, policy_params, _ = ppo.train(
        environment=env,
        eval_env=registry.load(env_name),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        progress_fn=progress,
        seed=seed,
        save_checkpoint_path=str(checkpoint_dir),
        **train_params,
    )
    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")

    # =========================================================================
    # 3. RENDER A ROLLOUT, THEN UPLOAD TO S3
    # =========================================================================
    # Upload even when rendering fails, so a training run never loses its
    # checkpoints; the render error still fails the job afterwards.
    try:
        write_rollout(
            env, make_inference_fn, policy_params, params.episode_length, seed, output_dir / "rollout.mp4"
        )
    finally:
        if bucket:
            upload_dir(output_dir, bucket, prefix)
        else:
            print("WARNING: CHECKPOINT_BUCKET not set. Outputs will be lost when the container exits.")

if __name__ == "__main__":
    main()
