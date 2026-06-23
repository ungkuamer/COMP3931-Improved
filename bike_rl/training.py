"""PPO/MaskablePPO training loop. See RECREATE_SPEC.md §3.7, §6.1."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from bike_rl.env import BikePathEnv

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext

logger = logging.getLogger(__name__)


def make_env(
    bike_graph: object,
    walk_graph: object,
    cfg: Config,
    run_context: RunContext,
    rank: int,
    seed: int,
    budget: float = 100_000.0,
) -> gym.Env[object, object]:
    """Construct, seed, reset, and ``Monitor``-wrap a single ``BikePathEnv``.

    The env is reset immediately with ``options={"budget": budget}`` so the
    episode starts with the configured budget (plan 004's contract — budget
    is passed via ``reset`` options, not ``Config``). The per-env seed is
    ``seed + rank`` so workers get distinct, reproducible seeds (§3.7).

    Args:
        bike_graph: The existing bike network (loaded once in the parent
            process — §6.1; passed to workers via the SubprocVecEnv thunk).
        walk_graph: The walkable network candidates are drawn from.
        cfg: Config (reward weights, cost factor, PPO hyperparameters).
        run_context: Per-run output context (stored on the env for logging;
            the env does not create directories).
        rank: Worker index (0-based); added to ``seed`` for per-env seeding.
        seed: Base seed (typically ``cfg.seed``).
        budget: Initial episode budget. Defaults to 100_000.0; the CLI
            (plan 007) passes the real ``--budget`` value.

    Returns:
        A ``Monitor``-wrapped ``BikePathEnv`` with the episode already reset.
    """
    env = BikePathEnv(
        cast("Any", bike_graph),
        cast("Any", walk_graph),
        cfg,
        run_context,
    )
    env.reset(seed=seed + rank, options={"budget": float(budget)})
    return cast(gym.Env[object, object], Monitor(env, filename=None))


def make_vec_env(
    bike_graph: object,
    walk_graph: object,
    cfg: Config,
    run_context: RunContext,
    n_envs: int,
    seed: int | None = None,
    budget: float = 100_000.0,
    vec_env_cls: type[VecEnv] = SubprocVecEnv,
    start_method: str | None = None,
) -> VecEnv:
    """Build a vectorised env over ``n_envs`` ``BikePathEnv`` workers.

    Each worker is a ``Monitor``-wrapped ``BikePathEnv`` built by
    :func:`make_env` with a distinct seed (``base_seed + rank``). The
    in-memory ``bike_graph`` / ``walk_graph`` are captured in each thunk's
    closure; ``SubprocVecEnv`` serialises the thunks to workers with
    ``cloudpickle``, so **no worker re-downloads OSM data** (RECREATE_SPEC
    §6.1 — the graphs are loaded once in the parent process by the CLI and
    passed in here).

    Args:
        bike_graph: The existing bike network (parent-loaded, §6.1).
        walk_graph: The walkable network.
        cfg: Config.
        run_context: Per-run output context.
        n_envs: Number of parallel envs (§3.7 default is ``cpu_count() - 1``;
            the CLI computes that — this function just takes the number).
        seed: Base seed; defaults to ``cfg.seed``. Each worker gets
            ``seed + rank``.
        budget: Initial episode budget passed to each env's ``reset``.
        vec_env_cls: ``SubprocVecEnv`` (default, spawns processes) or
            ``DummyVecEnv`` (in-process; used by tests to avoid spawning).
        start_method: ``SubprocVecEnv`` ``start_method`` (``"spawn"`` /
            ``"forkserver"`` / ``"fork"`` / ``None``). Ignored for
            ``DummyVecEnv``.

    Returns:
        A vectorised env ready to pass to :func:`train_model`.
    """
    base_seed = cfg.seed if seed is None else seed

    def _make_env_fn(rank: int) -> gym.Env[object, object]:
        return make_env(bike_graph, walk_graph, cfg, run_context, rank, base_seed, budget)

    def _thunk(rank: int) -> Callable[[], gym.Env[object, object]]:
        return lambda: _make_env_fn(rank)

    env_fns: list[Callable[[], gym.Env[object, object]]] = [_thunk(r) for r in range(n_envs)]
    if vec_env_cls is SubprocVecEnv:
        return SubprocVecEnv(env_fns, start_method=start_method)
    return DummyVecEnv(env_fns)


class TrainingProgressCallback(BaseCallback):
    """Callback that records every terminated episode's reward (RECREATE_SPEC §5.12).

    Reads the SB3 ``info["episode"]`` dict that ``BikePathEnv.step`` emits on
    termination (plan 004's contract) and appends ``info["episode"]["r"]`` to
    ``episode_rewards``. Because the env emits the summary *on termination*
    (not at the next ``reset``), the final episode of a rollout is recorded
    too — fixing the original §5.12 "last episode dropped" bug.

    Optionally displays a ``tqdm`` progress bar over ``total_timesteps``.

    Attributes:
        episode_rewards: List of per-episode total rewards (one entry per
            terminated episode, across all envs), in termination order.
        episode_lengths: List of per-episode lengths, parallel to
            ``episode_rewards``.
    """

    def __init__(
        self,
        total_timesteps: int,
        progress_bar: bool = True,
        verbose: int = 0,
    ) -> None:
        """Store config; lists are populated during training.

        Args:
            total_timesteps: Total timesteps the model will train for (used
                only for the tqdm bar total).
            progress_bar: If True, show a tqdm progress bar. Tests pass
                ``False`` to keep output deterministic.
            verbose: SB3 callback verbosity.
        """
        super().__init__(verbose=verbose)
        self._total_timesteps = total_timesteps
        self._progress_bar = progress_bar
        self._pbar: Any = None
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_training_start(self) -> None:
        """Create the tqdm bar (if enabled) once training starts."""
        super()._on_training_start()
        if self._progress_bar:
            from tqdm import tqdm

            self._pbar = tqdm(total=self._total_timesteps, desc="training", leave=False)

    def _on_step(self) -> bool:
        """Record any episode info from this step's per-env ``infos``; update the bar.

        Returns ``True`` to continue training (return ``False`` to stop early;
        this implementation never stops early).
        """
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
        if self._pbar is not None:
            self._pbar.update(self.model.num_timesteps - self._pbar.n)
        return True

    def _on_training_end(self) -> None:
        """Close the tqdm bar (if any)."""
        super()._on_training_end()
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def train_model(
    envs: VecEnv,
    cfg: Config,
    run_context: RunContext,
    total_timesteps: int,
    progress_callback: TrainingProgressCallback | None = None,
) -> MaskablePPO:
    """Train a ``MaskablePPO`` model on ``envs`` (RECREATE_SPEC §3.7, §5.4).

    Uses ``MaskableActorCriticPolicy`` so the env's ``action_masks()`` is
    consumed (§5.4 — invalid actions are never sampled). Hyperparameters come
    from ``cfg``. ``total_timesteps`` is rounded up to a multiple of
    ``ppo_n_steps * n_envs`` (§3.7 — one update batch's worth). A
    :class:`TrainingProgressCallback` records every terminated episode
    (§5.12), and a :class:`CheckpointCallback` writes into
    ``run_context.output_dir / "checkpoints"``.

    Args:
        envs: Vectorised env (from :func:`make_vec_env`).
        cfg: Config (PPO hyperparameters, seed, device).
        run_context: Per-run output context for checkpoint paths.
        total_timesteps: Requested timesteps; rounded up to
            ``ppo_n_steps * envs.num_envs``.
        progress_callback: Optional pre-built callback. If given, it is used
            in place of the internally-created one so the caller (the CLI)
            can read ``.episode_rewards`` after training to plot
            ``training_rewards.png`` (§9). If None, a fresh callback is
            created internally (existing behaviour, unchanged).

    Returns:
        The trained ``MaskablePPO`` model.
    """
    n_envs = envs.num_envs
    batch = cfg.ppo_n_steps * n_envs
    rounded = max(batch, int(math.ceil(total_timesteps / batch) * batch))
    logger.info(
        "training total_timesteps=%d rounded=%d (n_steps=%d n_envs=%d)",
        total_timesteps,
        rounded,
        cfg.ppo_n_steps,
        n_envs,
    )

    out_dir = run_context.ensure_output_dir()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_freq = max(rounded // 10, 1)

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=envs,
        learning_rate=cfg.ppo_learning_rate,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        n_epochs=cfg.ppo_n_epochs,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_gae_lambda,
        clip_range=cfg.ppo_clip_range,
        seed=cfg.seed,
        device=cfg.device,
        verbose=0,
    )
    progress_cb = (
        progress_callback
        if progress_callback is not None
        else TrainingProgressCallback(rounded, progress_bar=False)
    )
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(ckpt_dir),
        name_prefix="bike_path_ppo",
        verbose=0,
    )
    model.learn(total_timesteps=rounded, callback=[progress_cb, ckpt_cb])
    return model
