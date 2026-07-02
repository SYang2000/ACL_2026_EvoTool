# Evolved policies (Theta*)

Each `<benchmark>/theta_star.json` is the deployed best policy Theta* found by the
EvoTool loop on that benchmark: the four evolved prompt specifications
(planner / selector / caller / synthesizer) plus the policy id and run provenance.
These human-readable prompt specs ARE the method's trained artifact — the analogue of
model checkpoints for a prompt-evolution method — reconstructed from the run logs by
replaying the deployed policy's mutation chain from the initial specs.

Provenance caveat: these come from a reference testbed run (Qwen3-4B backbone,
90/30/30 split, seed 42, 3 epochs); the paper's reported numbers use Qwen3-8B.
Rerunning `python run.py --config configs/evotool.yaml --benchmark <ds>` regenerates
comparable artifacts (the deployed policy is logged in the run's result JSON and runlog).
