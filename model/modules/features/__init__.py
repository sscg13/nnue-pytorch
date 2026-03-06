import argparse
from collections.abc import Callable
from dataclasses import dataclass

from .composed import ComposedFeatureTransformer, combine_input_features
from .full_threatsv2 import FullThreatsv2
from .halfka_v2_hm import HalfKav2Hm
from .input_feature import InputFeature

import tyro
from typing import Annotated


_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "Full_Threatsv2": FullThreatsv2,
}


def get_feature_cls(name: str) -> Callable[[int], ComposedFeatureTransformer]:
    parts = name.split("+")
    components = [_FEATURE_COMPONENTS[p] for p in parts]
    return combine_input_features(*components)


def get_available_features() -> list[str]:
    return list(_FEATURE_COMPONENTS.keys())


@dataclass
class FeatureConfig:
    features: Annotated[
        str,
        tyro.conf.arg(
            help="The feature set to use. Available: "
            + ", ".join(get_available_features())
        ),
    ] = "HalfKAv2_hm^"


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default="Full_Threatsv2+HalfKAv2_hm^",
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threatsv2+HalfKAv2_hm^",
    )


__all__ = [
    "ComposedFeatureTransformer",
    "combine_input_features",
    "HalfKav2Hm",
    "FullThreatsv2",
    "InputFeature",
    "get_feature_cls",
    "get_available_features",
    "add_feature_args",
    "FeatureConfig",
]
