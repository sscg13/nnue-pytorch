from dataclasses import dataclass
from typing import Annotated, Literal

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    rule50: Literal["none", "ft", "hidden1", "hidden2"] = "none"
    """Discrete halfmove embedding placement (101 rows, clocks clamped to 0..100)."""

    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 32
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""
