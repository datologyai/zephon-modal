# Agent notes

Start with README.md (concepts, smoke test, commands, what's been tested),
then docs/design.md (the reconciler, preemption and sample hashes).

- `make`-free for now: `uv sync --group dev`, then `uv run pytest`,
  `uv run ruff format . && uv run ruff check .`, `uv run pyright`.
- Match zephon's conventions (Apache-2.0 SPDX headers, Google docstrings,
  ruff/pyright settings) so this package can move into the zephon repo later.
- Use only Zephon's public API, never `zephon._internal`. Needing an internal
  means Zephon needs a public API for it.
- Keep Datology-specific names (Modal workspace, Volume names, recipes) out
  of this repo; they live in zephon-modal-demo.
- Modal runs cost money and the workspace is shared: prefix run IDs and
  Volume names with your name, and prefer `examples/smoke` (T4s) over GPU
  recipes while iterating.
- `zephon_modal/zephon/torchtitan.py` and `megatron_entry.py` only import
  inside training images (torch, torchtitan, Megatron). The rest of the
  package must import without torch; `checkpoints.py` imports it lazily
  (it's a dev dependency for the tests).
