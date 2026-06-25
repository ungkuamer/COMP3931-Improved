# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# Workflow
- When dispatching plan executors, use isolated worktrees (`isolation: "worktree"`) per closing-the-loop.md. Confidence: 0.65

# Git
- When the user explicitly asks to merge a branch, merge it for them despite any default "never merge" policy. Confidence: 0.85
- After merging a branch, push to remote and clean up the merged branch locally without being separately prompted. Confidence: 0.70

