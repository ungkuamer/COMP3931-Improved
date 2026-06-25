"""Evaluation harness for solvers and RL policies. See OPTIMIZER_SPEC.md §8.

One runner for every solver (greedy / local-search / ILP) **and** the RL
policy, so the comparison is apples-to-apples: each is scored into the same
:class:`Solution` record by the **shared** :func:`bike_rl.objective.objective`,
never by the RL reward. This is what makes the RL number and the optimiser
number directly comparable (OPTIMIZER_SPEC §3.2 / §8).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import networkx as nx

from bike_rl.candidates import Candidate, extract_candidates
from bike_rl.objective import Edge, ObjectiveWeights, objective
from bike_rl.optim.greedy import GreedySolver, Solution

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


# ── Instance ──────────────────────────────────────────────────────────────


@dataclass
class Instance:
    """A problem instance shared by every solver in a comparison (§8).

    Attributes:
        graph: The base bike network graph (not mutated by solvers).
        candidates: Candidate edges to consider. Derived once from
            ``(graph, walk_graph, cfg)`` via :func:`extract_candidates` so
            every solver — including the RL env rollout — sees the same set.
        budget: Total budget; never exceeded.
        cfg: Config passed through to solvers and the objective.
        weights: Objective weights used to score every solution.
        walk_graph: The walkable network; required for RL policy rollouts
            (the env re-extracts candidates from ``(graph, walk_graph, cfg)``
            and must produce the same set as ``candidates``). May be ``None``
            when no RL row is evaluated.
        label: Human-readable instance name for reporting.
    """

    graph: _NXGraph
    candidates: list[Candidate]
    budget: float
    cfg: Config
    weights: ObjectiveWeights
    walk_graph: _NXGraph | None = None
    label: str = ""


# ── Solver protocol ───────────────────────────────────────────────────────


class _Solver(Protocol):
    """Structural type for anything with a ``solve(graph, candidates, budget)``."""

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution: ...


# ── Added-edge adapter (RL rollout → Edge protocol) ───────────────────────


@dataclass(frozen=True)
class _AddedEdge:
    """Adapter making an env-added ``(u, v, data)`` edge satisfy :class:`Edge`.

    The env records added edges as ``(u, v, data_dict)`` triples. The shared
    :func:`objective` consumes the :class:`~bike_rl.objective.Edge` protocol
    (``u``/``v``/``length``/``data``); this adapter bridges the two without
    depending on :class:`Candidate`'s identity fields (which are irrelevant to
    scoring). ``length`` is read from ``data`` (the env sets it on add).
    """

    u: int | str
    v: int | str
    length: float
    data: dict[str, Any]


# ── Core evaluators ───────────────────────────────────────────────────────


def evaluate_solver(solver: _Solver, instance: Instance) -> Solution:
    """Run ``solver`` on ``instance`` and return a comparable :class:`Solution`.

    Calls ``solver.solve(graph, candidates, budget)`` and then **re-scores**
    the returned edges with the shared :func:`objective` so the §8
    comparability guarantee holds regardless of which solver produced them:
    ``Solution.objective == objective(graph, sol.edges, weights, cfg)``.

    Args:
        solver: A solver exposing ``solve(graph, candidates, budget) -> Solution``.
        instance: The problem instance.

    Returns:
        A :class:`Solution` with the solver's own ``edges``/``spent``/
        ``runtime_s``/``solver``/``extra`` and an ``objective`` re-derived
        from the shared scorer.
    """
    start = time.perf_counter()
    sol = solver.solve(instance.graph, instance.candidates, instance.budget)
    eval_overhead = time.perf_counter() - start
    # §8 comparability: always re-score with the shared objective.
    rescored = objective(instance.graph, sol.edges, instance.weights, instance.cfg)
    return Solution(
        edges=list(sol.edges),
        objective=rescored,
        spent=float(sol.spent),
        runtime_s=float(sol.runtime_s),
        solver=sol.solver,
        extra={
            **dict(sol.extra),
            "eval_overhead_s": eval_overhead,
            "n_edges": len(sol.edges),
        },
    )


def evaluate_rl_policy(
    policy: object,
    instance: Instance,
    budget: float | None = None,
    seed: int | None = None,
    run_context: RunContext | None = None,
) -> Solution:
    """Run a trained RL policy deterministically and score it (§8).

    Performs one deterministic rollout of ``policy`` on a fresh
    :class:`~bike_rl.env.BikePathEnv` built from ``instance`` (using
    ``instance.walk_graph``), collects the chosen edges, and scores them with
    the **shared** :func:`objective` — the same scorer used by
    :func:`evaluate_solver` — so the RL row is directly comparable to the
    optimiser rows. The RL reward is **not** reported here (OPTIMIZER_SPEC
    §3.2: the reported metric is always the canonical objective).

    Args:
        policy: A trained ``MaskablePPO`` (typed ``object`` to avoid a hard
            runtime import of ``sb3_contrib``; only ``policy.predict`` is
            used, with ``deterministic=True`` and ``action_masks=…``).
        instance: The problem instance. ``instance.walk_graph`` must be set.
        budget: Override for ``instance.budget`` (episode budget passed via
            ``reset(options={"budget": …})``). Defaults to ``instance.budget``.
        seed: Reset seed; defaults to ``instance.cfg.seed``.
        run_context: Optional :class:`RunContext` for the env. If ``None``, a
            throwaway one is built (the env does not touch the filesystem
            during a pure rollout).

    Returns:
        A :class:`Solution` with ``solver="rl"``, the chosen edges, the
        shared-objective score, spent cost, rollout runtime, and ``extra``
        carrying ``n_edges``/``episode_reward``/``n_candidates``.

    Raises:
        ValueError: If ``instance.walk_graph`` is ``None`` (RL rollout needs it).
    """
    if instance.walk_graph is None:
        raise ValueError(
            "evaluate_rl_policy: instance.walk_graph is None; an RL rollout "
            "requires the walk graph so the env can build its action space."
        )

    # Local import to keep sb3_contrib / gymnasium out of the optimiser import
    # graph for pure-optimiser callers (greedy/local-search/ilp).
    from bike_rl.env import BikePathEnv
    from bike_rl.run_context import RunContext as _RunContext

    if run_context is None:
        # No filesystem side effect: RunContext.create only builds the
        # dataclass; ensure_output_dir() is never called during a rollout.
        run_context = _RunContext.create(Path("/tmp"), "eval_rl")

    start = time.perf_counter()
    env = BikePathEnv(instance.graph, instance.walk_graph, instance.cfg, run_context)
    ep_budget = float(instance.budget if budget is None else budget)
    obs, _ = env.reset(
        seed=instance.cfg.seed if seed is None else seed, options={"budget": ep_budget}
    )

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        if not bool(mask.any()):
            break  # no legal action — episode is over
        action, _ = policy.predict(obs, deterministic=True, action_masks=mask)  # type: ignore[attr-defined]
        obs, reward, terminated, truncated, _info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if steps > 10_000:  # hard safety cap; env terminates by budget normally
            logger.warning("evaluate_rl_policy: hit 10_000-step safety cap")
            break

    raw = env._graph  # noqa: SLF001 — read-only; mirrors bike_rl.evaluation
    added: list[_AddedEdge] = [
        _AddedEdge(
            u=u,
            v=v,
            length=float(d.get("length", 0.0)),
            data=dict(d),
        )
        for u, v, d in raw.edges(data=True)
        if d.get("bike_lane") == "yes" and not instance.graph.has_edge(u, v)
    ]
    runtime_s = time.perf_counter() - start

    spent = sum(float(d.data.get("length", 0.0)) for d in added) * instance.cfg.edge_cost_factor
    score = objective(instance.graph, cast("list[Edge]", added), instance.weights, instance.cfg)

    return Solution(
        edges=list(added),
        objective=score,
        spent=spent,
        runtime_s=runtime_s,
        solver="rl",
        extra={
            "n_edges": len(added),
            "n_candidates": instance.candidates and len(instance.candidates),
            "episode_reward": total_reward,
            "steps": steps,
        },
    )


# ── Comparison runner + §10 table ─────────────────────────────────────────


def default_solvers(instance: Instance) -> list[_Solver]:
    """Build the default solver roster for an instance (greedy, LS, ILP).

    Args:
        instance: The problem instance (provides ``cfg``/``weights``).

    Returns:
        ``[GreedySolver, LocalSearchSolver, ILPSolver]`` configured from the
        instance's ``cfg`` and ``weights``.
    """
    from bike_rl.optim.ilp import ILPSolver
    from bike_rl.optim.local_search import LocalSearchSolver

    return [
        GreedySolver(instance.cfg, instance.weights),
        LocalSearchSolver(instance.cfg, instance.weights),
        ILPSolver(instance.cfg, instance.weights),
    ]


def run_comparison(
    instance: Instance,
    solvers: list[_Solver] | None = None,
    rl_policy: object | None = None,
) -> list[Solution]:
    """Run every solver (+ optional RL) on ``instance`` and return their solutions.

    Args:
        instance: The problem instance.
        solvers: Solver roster; defaults to :func:`default_solvers`.
        rl_policy: Optional trained RL policy; when given, an ``"rl"`` row is
            appended via :func:`evaluate_rl_policy`.

    Returns:
        A list of :class:`Solution`, one per solver (RL row last when given),
        each scored by the shared :func:`objective`.
    """
    roster = default_solvers(instance) if solvers is None else solvers
    out = [evaluate_solver(s, instance) for s in roster]
    if rl_policy is not None:
        out.append(evaluate_rl_policy(rl_policy, instance))
    return out


def format_comparison_table(solutions: list[Solution], instance: Instance | None = None) -> str:
    """Render the OPTIMIZER_SPEC §10 reporting table as Markdown.

    Columns: Solver | Objective | Budget used | Runtime | # edges |
    Optimality gap | (ILP status). The optimality gap is measured against the
    ILP row when its status is ``OPTIMAL`` (the true ceiling); otherwise
    against the best objective in the table (labelled ``"vs best"``). With
    non-coverage-only weights the ILP optimises a coverage surrogate, not the
    full objective — see :class:`~bike_rl.optim.ilp.ILPSolver`; a footnote is
    emitted in that case.

    Args:
        solutions: Solutions from :func:`run_comparison`.
        instance: Optional instance; its ``label`` is used as the caption.

    Returns:
        A Markdown string.
    """
    if not solutions:
        return "(no solutions)"

    # Identify the ceiling: an OPTIMAL ILP row, else the best objective.
    ilp_rows = [s for s in solutions if s.solver == "ilp" and s.extra.get("status") == "OPTIMAL"]
    ceiling = ilp_rows[0].objective if ilp_rows else max(s.objective for s in solutions)
    ceiling_label = "vs ILP" if ilp_rows else "vs best"

    header = (
        "| Solver | Objective | Budget used | Runtime | # edges | "
        f"Optimality gap ({ceiling_label}) | ILP status |"
    )
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for s in solutions:
        gap = (ceiling - s.objective) / ceiling if ceiling > 0 else 0.0
        status = str(s.extra.get("status", "—"))
        rows.append(
            f"| {s.solver} | {s.objective:.4f} | {s.spent:.1f} | "
            f"{_fmt_runtime(s.runtime_s)} | {len(s.edges)} | {gap * 100:.2f}% | {status} |"
        )

    footnote = ""
    if ilp_rows and instance is not None and not _is_coverage_only(instance.weights):
        footnote = (
            "\n\n> Note: ILP status is OPTIMAL over its **coverage surrogate**, "
            "not the full weighted objective (weights are not coverage-only). "
            "For a true optimality ceiling, re-run with "
            "`ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)`."
        )
    caption = f"\n\n**Instance:** {instance.label or '(unnamed)'}" if instance else ""
    return "\n".join(rows) + caption + footnote


def _is_coverage_only(w: ObjectiveWeights) -> bool:
    """True iff weights are coverage-only (connectivity=0, fragmentation=0)."""
    return w.connectivity == 0.0 and w.fragmentation == 0.0 and w.coverage > 0.0


def _fmt_runtime(seconds: float) -> str:
    """Format a runtime in seconds compactly (``1.23s`` / ``2m04s`` / ``1h02m``)."""
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def build_instance_from_graphs(
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
    weights: ObjectiveWeights,
    budget: float,
    label: str = "",
) -> Instance:
    """Derive candidates once and build an :class:`Instance` for a comparison.

    Candidates are extracted from ``(bike_graph, walk_graph, cfg)`` so the
    optimisers and the RL env (which re-extracts the same way) see an
    identical candidate set — the fairness guarantee for §8.

    Args:
        bike_graph: The existing bike network.
        walk_graph: The walkable network candidates are drawn from.
        cfg: Config.
        weights: Objective weights.
        budget: Total budget.
        label: Optional instance label.

    Returns:
        An :class:`Instance` with ``candidates`` populated and ``walk_graph``
        set (ready for RL rollout).
    """
    return Instance(
        graph=bike_graph,
        candidates=extract_candidates(bike_graph, walk_graph, cfg),
        budget=budget,
        cfg=cfg,
        weights=weights,
        walk_graph=walk_graph,
        label=label,
    )


__all__ = [
    "Instance",
    "Solution",
    "build_instance_from_graphs",
    "default_solvers",
    "evaluate_rl_policy",
    "evaluate_solver",
    "format_comparison_table",
    "run_comparison",
]
