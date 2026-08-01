import chess
import torch
from torch import nn

from .input_feature import InputFeature


def _piece_index(is_white_pov: bool, piece: chess.Piece) -> int:
    """Map pieces to friendly pawn..king, then enemy pawn..king."""
    return (piece.piece_type - 1) + 6 * (piece.color != is_white_pov)


def _feature_index(
    is_white_pov: bool, king_sq: int, sq: int, piece: chess.Piece
) -> int:
    """P_hm index matching Stockfish's horizontally mirrored orientation."""
    horizontal_flip = 7 * ((king_sq % 8) < 4)
    perspective_flip = 56 * (not is_white_pov)
    oriented_sq = sq ^ horizontal_flip ^ perspective_flip
    return oriented_sq + _piece_index(is_white_pov, piece) * 64


class PHm(InputFeature):
    HASH = 0x6F234CB8
    FEATURE_NAME = "P_hm"
    INPUT_FEATURE_NAME = "P_hm"
    MAX_ACTIVE_FEATURES = 32

    NUM_INPUTS = 12 * 64
    NUM_REAL_FEATURES = NUM_INPUTS

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        scale = 1.0 / nnue2score
        l1_size = self.num_outputs - num_psqt_buckets
        initial_values = (
            torch.tensor(
                self._psqt_values(), device=self.weight.device, dtype=self.weight.dtype
            )
            * scale
        )

        for i in range(num_psqt_buckets):
            self.weight[:, l1_size + i] = initial_values

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    @staticmethod
    def _psqt_values() -> list[int]:
        piece_values = [126, 781, 825, 1276, 2538, 0]
        values = [0] * PHm.NUM_INPUTS

        for enemy in range(2):
            sign = -1 if enemy else 1
            for piece_type, value in enumerate(piece_values):
                offset = (enemy * 6 + piece_type) * 64
                values[offset : offset + 64] = [sign * value] * 64

        return values
