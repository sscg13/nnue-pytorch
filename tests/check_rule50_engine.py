"""Cross-check a nonzero embedding against an engine built for that placement.

python tests/check_rule50_engine.py --mode ft --baseline baseline.nnue --engine /path/to/stockfish
"""
import argparse
import gc
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import numpy as np
import torch

import data_loader
from ftperm import ft_permute_impl
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.utils.serialize import NNUEReader, NNUEWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["ft", "hidden1", "hidden2"])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", default="logs/rule50-crosscheck")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with open(args.baseline, "rb") as f:
        base = NNUEReader(f, DEFAULT_FEATURES, ModelConfig()).model
    model = NNUEModel(DEFAULT_FEATURES, ModelConfig(rule50=args.mode))
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert len(result.missing_keys) == 1 and not result.unexpected_keys, result
    del base
    gc.collect()
    table = model.input.rule50 if args.mode == "ft" else model.layer_stacks.rule50
    # Exercise every row, both signs, every output, and each material stack.
    with torch.no_grad():
        values = torch.arange(table.weight.numel()).reshape(table.weight.shape)
        table.weight.copy_(((values * 17 + values // 101) % 97 - 48) / 256.0)
        ft_permute_impl(model, np.arange(511, -1, -1))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    net = output_dir / f"{args.mode}.nnue"
    net.write_bytes(NNUEWriter(model, ft_compression="leb128", verbose=False).buf)

    boards = [chess.Board()]
    # All eight material stacks, both kings safely separated.
    for count in (3, 6, 10, 14, 18, 22, 26, 30):
        b = chess.Board()
        for square in list(b.piece_map()):
            if len(b.piece_map()) <= count:
                break
            if b.piece_at(square).piece_type != chess.KING:
                b.remove_piece_at(square)
        b.castling_rights = 0
        if b.is_valid() and not b.is_check():
            boards.append(b)
    boards.append(chess.Board("8/8/3k4/8/3K4/8/4R3/8 w - - 0 1"))
    fens = []
    for b in boards:
        for side in (chess.WHITE, chess.BLACK):
            b.turn = side
            if not b.is_valid() or b.is_check():
                continue
            for clock in range(101):
                b.halfmove_clock = clock
                fens.append(b.fen())
            b.halfmove_clock = 150
            fens.append(b.fen())
    native = data_loader.get_sparse_batch_from_fens(model.input_feature_name, fens, [0]*len(fens), [0]*len(fens), [0]*len(fens))
    try:
        us, them, w, b, _, _, pc, clock = native.contents.get_tensors("cpu")
    finally:
        data_loader.destroy_sparse_batch(native)
    with torch.no_grad():
        psqt, stacks = model.calculate_buckets(pc)
        transformed, white_psqt, black_psqt = model.forward_ft(us, them, w, b, psqt, True, True, rule50=clock)
        positional = model.layer_stacks(transformed, stacks, rule50=clock)
        # Match the engine's separate truncations of PSQT and positional output.
        # The usual floating trainer score can differ by almost two raw units.
        psqt_int = torch.round((white_psqt-black_psqt) * (us*2-1) * 9600).long()
        psqt_int = torch.div(torch.div(psqt_int, 2, rounding_mode="trunc"), 16, rounding_mode="trunc")
        pos_int = torch.div(torch.round(positional * 9600).long(), 16, rounding_mode="trunc")
        expected = (psqt_int + pos_int).flatten().numpy()
    commands = [f"setoption name EvalFile value {net}"]
    for fen in fens:
        commands += [f"position fen {fen}", "eval"]
    commands.append("quit")
    run = subprocess.run([str(Path(args.engine).resolve())], input="\n".join(commands)+"\n", capture_output=True, text=True, check=True)
    actual = np.array([int(x) for x in re.findall(r"NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)", run.stdout)])
    if len(actual) != len(fens):
        (output_dir / f"{args.mode}-engine.log").write_text(run.stdout + run.stderr)
        raise AssertionError(f"expected {len(fens)} evaluations, got {len(actual)}")
    errors = np.abs(expected-actual)
    print(f"{args.mode}: {len(fens)} positions, material stacks {sorted(set(((pc-1)//4).tolist()))}, max raw-eval error {errors.max():.6f}", flush=True)
    assert errors.max() == 0, (fens[errors.argmax()], expected[errors.argmax()], actual[errors.argmax()])
    # Bench exercises search, accumulator reuse, move/undo, and null moves.
    commands = f"setoption name EvalFile value {net}\nbench 16 1 5 default depth\nquit\n"
    run = subprocess.run([str(Path(args.engine).resolve())], input=commands, capture_output=True, text=True, check=True)
    assert "Nodes searched" in run.stderr + run.stdout
    (output_dir / f"{args.mode}-bench.log").write_text(run.stdout + run.stderr)
    print(f"{args.mode}: depth-5 search bench passed", flush=True)


if __name__ == "__main__":
    main()
