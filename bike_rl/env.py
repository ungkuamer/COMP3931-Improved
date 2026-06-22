"""Gymnasium RL environment for budgeted bike-network expansion.

See RECREATE_SPEC.md §3.4 (behaviour), §5.4 (action masking),
§5.5 (budget-efficiency reward), §5.6 (state updates), §5.11 (logging),
§5.12 (episode bookkeeping), §5.14 (cache versioning via MetricsState).

The env exposes ``action_masks()`` so ``sb3-contrib``'s ``MaskablePPO``
(plan 005) can mask invalid (already-used or unaffordable) actions over a
**fixed-size** action space. This module does **not** import ``sb3_contrib``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import networkx as nx
import numpy as np
import numpy.typing as npt

from bike_rl.candidates import Candidate, candidate_cost, extract_candidates
from bike_rl.metrics import MetricsState

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


class BikePathEnv(gym.Env[object, object]):
    """Bike-path expansion env with MaskablePPO action masking (RECREATE_SPEC §3.4/§5.4).

    Observation: ``Box(0, 1, (4,), float32)`` =
    ``[connectivity, path_efficiency, coverage, budget/initial_budget]``.
    Action space: ``Discrete(n_initial_candidates)`` — **fixed** for the life
    of the instance (§5.4). ``action_masks()`` returns a boolean mask over
    that fixed space; ``True`` = the candidate at that index is still unused
    *and* affordable with the remaining budget. Invalid/masked actions yield
    reward 0 and do **not** terminate the episode (§5.4).
    """

    def __init__(
        self,
        bike_graph: _NXGraph,
        walk_graph: _NXGraph,
        cfg: Config,
        run_context: RunContext,
    ) -> None:
        """Store inputs and build the fixed action space; call :meth:`reset` is NOT done here.

        Args:
            bike_graph: The existing bike network (edges with ``bike_lane='yes'``).
                Copied internally; not mutated.
            walk_graph: The walkable network candidates are drawn from.
                Not mutated (§5.8 — ``extract_candidates`` copies edge data).
            cfg: Config (reward weights, cost factor, step cap, seed).
            run_context: Per-run output context (stored for logging/future
                use by evaluation; the env does not create directories).
        """
        super().__init__()
        self._bike_graph_original = bike_graph.copy()
        self._walk_graph = walk_graph
        self._cfg = cfg
        self._run_context = run_context

        # Fixed action space = initial candidate count (padded to ≥1) — §5.4.
        initial_candidates = extract_candidates(bike_graph, walk_graph, cfg)
        self._n_actions = max(1, len(initial_candidates))
        self.action_space = gym.spaces.Discrete(self._n_actions)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Episode state (initialised properly in reset).
        self._initial_budget: float = 0.0
        self._budget: float = 0.0
        self._steps: int = 0
        self._graph: _NXGraph = nx.MultiDiGraph()
        self._metrics: MetricsState | None = None
        self._slots: list[Candidate | None] = []
        self._bike_nodes: set[int | str] = set()
        self._bike_nodes_before: set[int | str] = set()
        self._last_obs: npt.NDArray[np.float32] = np.zeros(4, dtype=np.float32)
        self._last_frag: float = 0.0
        self._episode_reward: float = 0.0

    # ── Episode lifecycle ────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the episode: fresh graph copy, full budget, all slots available.

        Args:
            seed: Optional seed (passed to ``super().reset``; the env is
                otherwise deterministic given ``cfg``).
            options: Optional dict; recognised key ``"budget"`` (float)
                overrides the default initial budget. The CLI/training
                (plan 005/007) passes the budget via options.

        Returns:
            ``(observation, info)`` with an empty info dict.
        """
        super().reset(seed=seed)
        cfg = self._cfg
        self._graph = self._bike_graph_original.copy()
        self._metrics = MetricsState(self._graph, cfg)
        candidates = extract_candidates(self._bike_graph_original, self._walk_graph, cfg)
        # Pad slots to the fixed action-space size with None.
        self._slots = list(candidates) + [None] * (self._n_actions - len(candidates))
        self._bike_nodes = {
            n
            for u, v, d in self._graph.edges(data=True)
            if d.get("bike_lane") == "yes"
            for n in (u, v)
        }
        if options is not None and "budget" in options:
            self._initial_budget = float(options["budget"])
        if self._initial_budget <= 0.0:
            # Fallback when no budget has been configured: a large default so
            # tiny fixtures still terminate via candidate exhaustion. The
            # CLI always sets a real budget (plan 007).
            self._initial_budget = 100_000.0
        self._budget = self._initial_budget
        self._steps = 0
        self._episode_reward = 0.0
        self._last_obs = self._get_observation()
        self._last_frag = self._metrics.fragmentation()
        return self._last_obs.copy(), {}

    def _get_observation(self) -> npt.NDArray[np.float32]:
        """Return ``[connectivity, efficiency, coverage, budget/initial]`` as float32.

        All four components are in ``[0, 1]``. Reads from ``self._metrics``
        (incremental + version-cached — §5.6/§5.14) so a full state read is
        cheap and always consistent.
        """
        assert self._metrics is not None
        conn = self._metrics.connectivity()
        eff = self._metrics.path_efficiency()
        pop = self._metrics.coverage()
        norm_budget = self._budget / self._initial_budget if self._initial_budget > 0 else 0.0
        norm_budget = max(0.0, min(1.0, norm_budget))
        obs: npt.NDArray[np.float32] = np.asarray([conn, eff, pop, norm_budget], dtype=np.float32)
        return obs

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of shape ``(action_space.n,)``: legal = unused & affordable.

        ``True`` at index ``i`` iff ``self._slots[i]`` is a candidate that has
        not been used yet and whose cost fits the remaining budget. This is
        the MaskablePPO contract (§5.4).
        """
        cfg = self._cfg
        mask = np.zeros(self._n_actions, dtype=bool)
        for i, slot in enumerate(self._slots):
            if slot is None:
                continue
            if candidate_cost(slot, cfg) <= self._budget + 1e-9:
                mask[i] = True
        return mask

    # ── Step ─────────────────────────────────────────────────────────────

    def step(
        self, action: object
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one bike-lane addition (RECREATE_SPEC §3.4/§3.5/§5.4/§5.5/§5.6/§5.12).

        - If ``action`` is masked (used or unaffordable): reward 0, no state
          change, episode continues (§5.4).
        - Otherwise: add the edge to the graph and to ``MetricsState``,
          deduct cost, recompute the full state from ``MetricsState`` (§5.6 —
          always full, cheap via incremental structures + version cache
          §5.14), and compute the reward.
        - ``terminated`` when no action is legal afterwards (budget exhausted
          or all candidates used). ``truncated`` when ``max_episode_steps >
          0`` and the step cap is hit.
        - On termination, emit SB3-style episode info in ``info["episode"]``
          so the training callback (plan 005) records every episode,
          including the last (§5.12).
        """
        cfg = self._cfg
        assert self._metrics is not None

        mask = self.action_masks()
        # ── Invalid / masked action (§5.4): no-op, reward 0, not done ──
        if isinstance(action, np.integer):
            action_int = int(action)
        elif isinstance(action, int):
            action_int = action
        else:
            action_int = -1
        if action_int < 0 or action_int >= self._n_actions or not bool(mask[action_int]):
            logger.debug("masked action %s ignored", action)
            obs = self._get_observation().copy()
            info: dict[str, Any] = {"invalid_action": True}
            truncated = self._truncated_now()
            # An all-masked state should already have terminated last step;
            # but if the env is driven into a no-legal-action state, terminate now.
            terminated = not bool(mask.any())
            if terminated:
                info = {**info, **self._episode_info()}
            else:
                self._steps += 1
            return obs, 0.0, terminated, truncated, info

        candidate = self._slots[action_int]
        assert candidate is not None
        cost = candidate_cost(candidate, cfg)

        # ── State before ──
        old_obs = self._last_obs
        old_frag = self._last_frag

        # ── Apply the edge to the env graph AND to MetricsState ──
        self._apply_edge(candidate)
        self._budget -= cost

        # ── State after (full read — §5.6; cheap via MetricsState — §5.14) ──
        new_obs = self._get_observation()
        new_frag = self._metrics.fragmentation()

        # ── Reward (§3.5 structure + §5.5 fix) ──
        wc, we, wp = cfg.reward_weights()
        state_gain = (
            (float(new_obs[0]) - float(old_obs[0])) * wc
            + (float(new_obs[1]) - float(old_obs[1])) * we
            + (float(new_obs[2]) - float(old_obs[2])) * wp
        )
        connects = (candidate.u in self._bike_nodes_before) or (
            candidate.v in self._bike_nodes_before
        )
        would_create_isolated = (
            candidate.u not in self._bike_nodes_before
            and candidate.v not in self._bike_nodes_before
        )

        reward = state_gain * cfg.reward_scale
        reward += candidate.road_priority * cfg.road_priority_scale
        if connects:
            reward += cfg.continuity_bonus
        reward += (old_frag - new_frag) * cfg.fragmentation_weight
        if would_create_isolated:
            reward += cfg.isolation_penalty  # negative (§3.5)

        # Budget-efficiency term (§5.5 fix — non-zero for positive gain + cost).
        gain_scaled = max(0.0, state_gain) * cfg.reward_scale
        if cost > 0 and self._initial_budget > 0:
            frac = cost / self._initial_budget
            budget_term = cfg.w_budget_efficiency * gain_scaled / frac
            reward += min(budget_term, cfg.budget_efficiency_cap)

        # ── Bookkeeping ──
        self._last_obs = new_obs
        self._last_frag = new_frag
        self._episode_reward += reward
        self._steps += 1

        terminated = not bool(self.action_masks().any())
        truncated = self._truncated_now()
        info_out: dict[str, Any] = {}
        if terminated:
            info_out = self._episode_info()
        return new_obs.copy(), float(reward), terminated, truncated, info_out

    def _apply_edge(self, candidate: Candidate) -> None:
        """Add the candidate edge to the env graph + MetricsState; update bike_nodes.

        Records ``connects_to_bike_path`` from the bike-node set *before* the
        add (stored on ``self._bike_nodes_before``) so the reward's continuity
        / isolation tests see the pre-add network state (§5.8/§3.5).
        """
        assert self._metrics is not None
        self._bike_nodes_before = set(self._bike_nodes)
        u, v = candidate.u, candidate.v
        data = dict(candidate.data)
        data["bike_lane"] = "yes"
        data["length"] = candidate.length
        self._graph.add_edge(u, v, **data)
        self._metrics.add_edge(u, v, data)
        # Ensure both endpoints exist as nodes in the env graph (they should
        # already, from the walk graph; this is a no-op safety).
        for n in (u, v):
            self._bike_nodes.add(n)
        # Mark the slot used (fixed action-space index — §5.4).
        idx = self._slots.index(candidate)
        self._slots[idx] = None

    def _truncated_now(self) -> bool:
        """True iff the step cap is enabled and reached."""
        return self._cfg.max_episode_steps > 0 and self._steps >= self._cfg.max_episode_steps

    def _episode_info(self) -> dict[str, Any]:
        """SB3-style episode summary for the ``info`` dict (§5.12)."""
        return {"episode": {"r": float(self._episode_reward), "l": int(self._steps)}}
