"""Add a zero-initialized discrete clock embedding to a baseline net for fine-tuning."""
import argparse
from dataclasses import replace

import torch

from model.config import ModelConfig, NNUELightningConfig
from model.modules.features import add_feature_args
from model.nnue import NNUE
from model.utils.load_model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Baseline .nnue, .ckpt or .pt")
    parser.add_argument("target", help="Output .pt to pass to train.py --resume-from-model")
    add_feature_args(parser)
    ModelConfig.add_model_args(parser)
    parser.add_argument("--l3", type=int, default=32)
    args = parser.parse_args()
    if args.rule50 == "none" or not args.target.endswith(".pt"):
        parser.error("Choose --rule50 ft/hidden1/hidden2 and a .pt target")
    config = ModelConfig.get_model_config(args)
    config.L3 = args.l3
    baseline = load_model(args.source, args.features, replace(config, rule50="none"))
    if getattr(baseline, "rule50", "none") != "none":
        parser.error("Source must be a baseline without a clock embedding")
    result = NNUE(NNUELightningConfig(features=args.features, model_config=config))
    loaded = result.model.load_state_dict(baseline.state_dict(), strict=False)
    expected = "input.rule50.weight" if args.rule50 == "ft" else "layer_stacks.rule50.weight"
    if loaded.missing_keys != [expected] or loaded.unexpected_keys:
        raise ValueError(f"Incompatible baseline: {loaded}")
    torch.save(result, args.target)
    print(f"Created {args.target}: {args.rule50}, 12 zero-initialized clock buckets")


if __name__ == "__main__":
    main()
