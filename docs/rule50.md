# Discrete halfmove embeddings

Select `--rule50 ft`, `--rule50 hidden1`, or `--rule50 hidden2` in training,
serialization, and cross-check commands. Default `none` preserves the baseline
network format. Every variant uses 101 independent learned rows indexed by
`clamp(halfmove_clock, 0, 100)`; there is no scalar projection or interpolation.
The clock is the fifth FEN field, not game ply.

| Mode | Placement | Parameters at 1024/32/32 | Uncompressed table |
|---|---|---:|---:|
| ft | Both perspective accumulations, before clipping/product | 103,424 | 202 KiB, int16 |
| hidden1 | First 32-unit preactivation, including the direct skip units | 25,856 | 101 KiB, int32 |
| hidden2 | Second 32-unit preactivation | 25,856 | 101 KiB, int32 |

FT shares one table across perspectives. Hidden variants have separate tables for
each of the eight material stacks. Tables initialize to zero. FT rows use scale
256 and are bounded to +/-127 integer units; hidden rows use destination bias
scales (16384 or 8192) and a conservative +/-2^20 integer bound. Bounds are applied
in forward/export with straight-through quantization inside the valid range.

The native batch ABI is version 2 and adds a final `rule50` tensor. Rebuild the
loader before using this branch; Python rejects old or incompatible libraries.
All standard batch consumers, including DDP and FT permutation, receive the
clock. FT training prepends the selected row's index before board indices because
the fused kernels stop at the first -1 padding entry. No custom kernel changes
are needed. Engine FT additions happen during transformation, with board
accumulators and Finny caches unchanged.

## Start from a baseline

```
python initialize_rule50.py baseline.nnue rule50-ft.pt --rule50 ft
python train.py data.binpack --resume-from-model rule50-ft.pt --rule50 ft
python serialize.py checkpoint.ckpt experiment.nnue --rule50 ft
```

Supply matching feature names and layer sizes for nondefault architectures. The
initializer accepts baseline `.nnue`, `.ckpt`, or `.pt` sources and writes a `.pt`
with zero clock weights. It rejects unexpected missing parameters. Resuming a
model requires a matching `--rule50` option. Existing pre-experiment pickled `.pt`
objects should be passed through the initializer when adding the embedding.

## Matching Stockfish builds

```
make -j4 rule50-ft-build ARCH=native
make -j4 rule50-hidden1-build ARCH=native
make -j4 rule50-hidden2-build ARCH=native
```

Each target cleans object files before changing architecture. They are ordinary
optimized builds, not PGO builds: the shipped baseline net has a different hash
and cannot profile the experimental engine. Select the corresponding experimental
net with UCI `setoption name EvalFile value ...` before evaluation/search. All three
formats have distinct signatures and reject mismatched nets. The existing
rule-50 evaluation damping is preserved.

## Validation completed

- Native loader build, clock transfer/ownership checks, independent-row and
  clamping checks, gradients, optimizer registration/updates, zero-init baseline
  equivalence, FT permutation, compressed/uncompressed serialization and wrong
  architecture rejection passed. Six existing DDP loader unit tests passed.
- Each AVX2 engine matched an integer-rounded trainer reference exactly on 2040
  positions spanning all 101 rows, both sides, and all eight material stacks, using
  nonzero embedding weights. Each passed a depth-5 search bench.
- 18440 comparisons of incremental/lazy, reused-cache, null-move and undo
  evaluations against fresh evaluation passed for hidden2 AVX2 and FT scalar.
  Positions include captures, pawn moves, castling, en passant and promotions.
- A short compiled CPU training/validation run passed for hidden1.
- CUDA/CuPy parity is implemented as an optional test but was skipped because the
  available environment has CPU-only PyTorch. The FT index placement was then
  corrected to precede padding based on inspection of the fused kernel; full
  numerical checks above predate that order-only correction.
- The two-process training attempt encountered the bundled one-chunk binpack's
  sharding failure. A multichunk rerun was prepared but execution was blocked by
  the account usage limit. Distributed training remains unverified end to end.

No full training or Elo experiment was run. Tests and entry points:

```
python -m unittest discover -s tests -p test_rule50.py -v
python tests/check_rule50_engine.py --mode ft --baseline baseline.nnue --engine /path/to/engine
# From Stockfish src, after building the same architecture and RULE50_LAYER:
make rule50-cache-test ARCH=general-64 RULE50_LAYER=1
./rule50-cache-test /absolute/path/to/ft.nnue
```

Use the same encoding and training recipe when comparing placements. Examine
coverage and validation loss by clock range before expensive games; high clock
rows may have relatively few examples. Change external rule-50 damping only in
a separate experiment.
