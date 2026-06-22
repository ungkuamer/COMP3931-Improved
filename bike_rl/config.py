"""Central configuration — all magic numbers in one frozen dataclass.

Every component reads its parameters from Config so that there is exactly one
source of truth for reward weights, thresholds, PPO hyperparameters, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Central configuration for bike-path RL and optimiser experiments.

    Attributes:
        w_connectivity: Reward weight for connectivity.
        w_efficiency: Reward weight for path efficiency.
        w_population: Reward weight for population coverage.
        reward_scale: Scaling factor for total reward.
        road_priority_scale: Scaling factor for road-priority bonus.
        continuity_bonus: Bonus for continuing an existing bike path.
        fragmentation_weight: Weight penalising network fragmentation.
        isolation_penalty: Negative penalty for isolated components.
        edge_cost_factor: Cost multiplier per edge length.
        min_candidate_length: Minimum candidate edge length (metres).
        max_candidate_length: Maximum candidate edge length (metres).
        road_priorities: Mapping from highway type to priority score.
        coverage_radius_m: Radius (metres) for population coverage proxy.
        coverage_mode: Method for coverage calculation ("radius" etc.).
        sampling_thresholds: Node-count thresholds for graph-size categories.
        clustering_sample_sizes: Sample sizes for clustering in large/huge graphs.
        path_sample_formula: Formula identifier for path sampling ("log" etc.).
        update_freq_huge: Update frequency for huge graphs.
        update_freq_large: Update frequency for large graphs.
        update_freq_default: Update frequency for other graphs.
        ppo_policy: PPO policy network type (e.g. "MlpPolicy").
        ppo_learning_rate: PPO learning rate.
        ppo_n_steps: PPO steps per update.
        ppo_batch_size: PPO batch size.
        ppo_n_epochs: PPO epochs per update.
        ppo_gamma: PPO discount factor.
        ppo_gae_lambda: PPO GAE lambda.
        ppo_clip_range: PPO clip range.
        device: Torch device ("cpu" or "cuda").
        local_search_max_iter: Max iterations for local search.
        local_search_time_limit_s: Time limit (seconds) for local search.
        ilp_time_limit_s: Time limit (seconds) for ILP solver.
        ilp_default_solver: Default ILP solver ("ortools").
        max_candidates_for_ilp: Max candidates before ILP applies filtering.
        seed: Random seed for reproducibility.
    """

    # Reward weights
    w_connectivity: float = 0.9
    w_efficiency: float = 0.2
    w_population: float = 0.3
    reward_scale: float = 100.0
    road_priority_scale: float = 5.0
    continuity_bonus: float = 50.0
    fragmentation_weight: float = 200.0
    isolation_penalty: float = -100.0

    # Cost / budget
    edge_cost_factor: float = 10.0

    # Candidate filtering
    min_candidate_length: float = 100.0
    max_candidate_length: float = 1000.0
    road_priorities: dict[str, int] = field(
        default_factory=lambda: {
            "primary": 5,
            "secondary": 4,
            "tertiary": 3,
            "residential": 2,
            "unclassified": 1,
        }
    )

    # Population-served proxy
    coverage_radius_m: float = 300.0
    coverage_mode: str = "radius"

    # Metrics sampling thresholds
    sampling_thresholds: dict[str, int] = field(
        default_factory=lambda: {
            "huge": 10000,
            "large": 5000,
            "medium": 1000,
            "small": 100,
        }
    )
    clustering_sample_sizes: dict[str, int] = field(
        default_factory=lambda: {"huge": 40, "large": 60}
    )
    path_sample_formula: str = "log"

    # Update frequency
    update_freq_huge: int = 5
    update_freq_large: int = 3
    update_freq_default: int = 1

    # PPO hyperparameters
    ppo_policy: str = "MlpPolicy"
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    device: str = "cpu"

    # Optimiser thresholds
    local_search_max_iter: int = 1000
    local_search_time_limit_s: float = 60.0
    ilp_time_limit_s: float = 300.0
    ilp_default_solver: str = "ortools"
    max_candidates_for_ilp: int = 300

    # Reproducibility
    seed: int = 0

    def reward_weights(self) -> tuple[float, float, float]:
        """Return (connectivity, efficiency, population) reward weights."""
        return (self.w_connectivity, self.w_efficiency, self.w_population)
