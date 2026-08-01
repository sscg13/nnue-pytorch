import chess

import data_loader
from model.modules.features.p_hm import PHm, _feature_index


def test_p_hm_contract():
    feature = PHm(num_outputs=16)

    assert feature.HASH == 0x6F234CB8
    assert feature.NUM_INPUTS == 768
    assert feature.NUM_REAL_FEATURES == 768
    assert feature.get_export_weights().data_ptr() == feature.weight.data_ptr()


def test_p_hm_orientation_and_piece_order():
    own_knight = chess.Piece(chess.KNIGHT, chess.WHITE)
    enemy_knight = chess.Piece(chess.KNIGHT, chess.BLACK)

    # Horizontal reflections are equivalent when the friendly king is reflected too.
    assert _feature_index(True, chess.C1, chess.B3, own_knight) == _feature_index(
        True, chess.F1, chess.G3, own_knight
    )

    # A 180-degree rotation with colors swapped is equivalent from Black's perspective.
    assert _feature_index(True, chess.C1, chess.B3, own_knight) == _feature_index(
        False, chess.F8, chess.G6, enemy_knight
    )

    # Friendly piece planes precede all enemy piece planes.
    assert _feature_index(True, chess.E1, chess.A1, own_knight) // 64 == 1
    assert _feature_index(True, chess.E1, chess.A1, enemy_knight) // 64 == 7


def test_native_p_hm_indices_match_python():
    fen = "4k3/7p/8/3n4/2B5/8/P7/3K4 w - - 0 1"
    board = chess.Board(fen)
    batch = data_loader.get_sparse_batch_from_fens("P_hm", [fen], [0], [1], [0])

    try:
        _, _, white_indices, black_indices, *_ = batch.contents.get_tensors("cpu")
        actual_white = set(white_indices[0][white_indices[0] >= 0].tolist())
        actual_black = set(black_indices[0][black_indices[0] >= 0].tolist())

        expected_white = {
            _feature_index(True, board.king(chess.WHITE), sq, piece)
            for sq, piece in board.piece_map().items()
        }
        expected_black = {
            _feature_index(False, board.king(chess.BLACK), sq, piece)
            for sq, piece in board.piece_map().items()
        }

        assert actual_white == expected_white
        assert actual_black == expected_black
    finally:
        data_loader.destroy_sparse_batch(batch)
