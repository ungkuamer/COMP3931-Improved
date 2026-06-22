"""bike_rl: RL and direct-optimiser baselines for budgeted bike-network expansion.

Shared modules (candidates, metrics, objective, graph_utils) are consumed by
both the Gymnasium RL environment and the direct optimisers (greedy / local
search / ILP) so the two approaches solve exactly the same problem.
"""

__version__ = "0.1.0"
