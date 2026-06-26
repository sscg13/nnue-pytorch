#include "training_data_loader_internal.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <future>
#include <random>
#include <cstring>
#include <cmath>

#include "lib/rng.h"

using namespace binpack;
using namespace chess;

// ---------------------------------------------------------
// Internal extractors and threat arrays
// ---------------------------------------------------------

static Square orient_flip_2(Color color, Square sq, Square ksq) {
    bool h = ksq.file() < fileE;
    if (color == Color::Black)
        sq = sq.flippedVertically();
    if (h)
        sq = sq.flippedHorizontally();
    return sq;
}

struct HalfKAv2_hm {
    static constexpr std::string_view NAME = "HalfKAv2_hm";

    static constexpr int NUM_SQ              = 64;
    static constexpr int NUM_PT              = 12;
    static constexpr int NUM_PLANES          = NUM_SQ * NUM_PT;
    static constexpr int INPUTS              = NUM_PLANES * NUM_SQ / 2;
    static constexpr int MAX_ACTIVE_FEATURES = 32;

    // clang-format off
    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };
    // clang-format on

    static int feature_index(Color color, Square ksq, Square sq, Piece p) {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto   p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ
             + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos    = e.pos;
        auto  pieces = pos.piecesBB();
        auto  ksq    = pos.kingSquare(color);

        int j = 0;
        for (Square sq : pieces)
        {
            auto p      = pos.pieceAt(sq);
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }
        return {j, INPUTS};
    }
};

struct HalfKAv2_hmExtractor: IFeatureExtractor {
    int inputs() const override { return HalfKAv2_hm::INPUTS; }
    int max_active_features() const override { return HalfKAv2_hm::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             Color                    color) const override {
        return HalfKAv2_hm::fill_features_sparse(e, features, color);
    }
};

// XOR-spacing-8 layout: W_ values are 0-5, B_ values are 8-13 (gaps at 6,7 and 14,15).
// Perspective flip is then a single XOR: at ^ (perspective<<3) or tt ^ (perspective<<3).
enum AttackType : int {
    W_PAWN_DIAG_AT = 0, W_PAWN_PAIR_AT = 1,
    W_KNIGHT_AT    = 2, W_BISHOP_AT    = 3, W_ROOK_AT  = 4, W_QUEEN_AT  = 5,
    // values 6,7 are unused gaps
    B_PAWN_DIAG_AT = 8, B_PAWN_PAIR_AT = 9,
    B_KNIGHT_AT    = 10, B_BISHOP_AT   = 11, B_ROOK_AT = 12, B_QUEEN_AT = 13,
    // values 14,15 are unused gaps
    ATTACK_TYPE_NB = 14
};

// XOR-spacing-8 layout: W_ values are 0-4, B_ values are 8-12 (gaps at 5-7).
enum TargetType : int {
    W_PAWN_TT = 0, W_KNIGHT_TT = 1, W_BISHOP_TT = 2, W_ROOK_TT = 3, W_QUEEN_TT = 4,
    // values 5-7 are unused gaps
    B_PAWN_TT = 8, B_KNIGHT_TT = 9, B_BISHOP_TT = 10, B_ROOK_TT = 11, B_QUEEN_TT = 12,
    TARGET_TYPE_NB = 13
};

// p.type(): Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
// p.color(): White=0, Black=1
// Base is 0 for White, 8 for Black — supports XOR perspective flip (at ^ (perspective<<3)).
constexpr AttackType make_attack_type(Piece p) {
    int base = (int) p.color() * 8;
    if (p.type() == PieceType::Pawn)
        return AttackType(base + 0);  // pawns always map to the diagonal AttackType
    return AttackType(base + (int) p.type() + 1);
}

constexpr TargetType make_target_type(Piece p) {
    // Caller must guarantee p.type() != King
    // Base is 0 for White, 8 for Black — supports XOR perspective flip (tt ^ (perspective<<3)).
    return TargetType((int) p.color() * 8 + (int) p.type());
}

// clang-format off
// Slot index for each (AttackType, TargetType) pair.
// -1 = fully excluded. >=0 = contiguous slot index used for feature base offset.
// The gap AT rows (6,7 and 14,15) and gap TT columns (5,6,7) are all -1.
// PAWN_DIAG does not target pawns; pawn-on-pawn co-presence is captured by PP_3Wide.
constexpr int8_t slot_map[ATTACK_TYPE_NB][TARGET_TYPE_NB] = {
  //                  W_P  W_N  W_B  W_R  W_Q  g5   g6   g7   B_P  B_N  B_B  B_R  B_Q
  /* W_PAWN_DIAG */ { -1,   0,  -1,   1,  -1,  -1,  -1,  -1,  -1,   2,  -1,   3,  -1},
  /* W_PAWN_PAIR */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
  /* W_KNIGHT    */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
  /* W_BISHOP    */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
  /* W_ROOK      */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
  /* W_QUEEN     */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
  /* gap_6       */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
  /* gap_7       */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
  /* B_PAWN_DIAG */ { -1,   0,  -1,   1,  -1,  -1,  -1,  -1,  -1,   2,  -1,   3,  -1},
  /* B_PAWN_PAIR */ { -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
  /* B_KNIGHT    */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
  /* B_BISHOP    */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
  /* B_ROOK      */ {  0,   1,   2,   3,  -1,  -1,  -1,  -1,   4,   5,   6,   7,  -1},
  /* B_QUEEN     */ {  0,   1,   2,   3,   4,  -1,  -1,  -1,   5,   6,   7,   8,   9},
};

// Semi-exclusion: true = only active when from_oriented >= to_oriented (FROM_GT convention).
constexpr bool semi_map[ATTACK_TYPE_NB][TARGET_TYPE_NB] = {
  //                  W_P    W_N    W_B    W_R    W_Q    g5     g6     g7     B_P    B_N    B_B    B_R    B_Q
  /* W_PAWN_DIAG */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
  /* W_PAWN_PAIR */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
  /* W_KNIGHT    */ {false,  true, false, false, false, false, false, false, false,  true, false, false, false},
  /* W_BISHOP    */ {false, false,  true, false, false, false, false, false, false, false,  true, false, false},
  /* W_ROOK      */ {false, false, false,  true, false, false, false, false, false, false, false,  true, false},
  /* W_QUEEN     */ {false, false, false, false,  true, false, false, false, false, false, false, false,  true},
  /* gap_6       */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
  /* gap_7       */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
  /* B_PAWN_DIAG */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
  /* B_PAWN_PAIR */ {false, false, false, false, false, false, false, false, false, false, false, false, false},
  /* B_KNIGHT    */ {false,  true, false, false, false, false, false, false, false,  true, false, false, false},
  /* B_BISHOP    */ {false, false,  true, false, false, false, false, false, false, false,  true, false, false},
  /* B_ROOK      */ {false, false, false,  true, false, false, false, false, false, false, false,  true, false},
  /* B_QUEEN     */ {false, false, false, false,  true, false, false, false, false, false, false, false,  true},
};
// clang-format on

// Pawn-pair geometry: own + adjacent files, restricted to ranks 2-7, excluding
// the source square. Color-independent. Mirrors Stockfish's pawn_pair_bb().
constexpr std::uint64_t pawn_pair_mask(int sq) {
    constexpr std::uint64_t FileA = 0x0101010101010101ULL;
    constexpr std::uint64_t Rank1 = 0xFFULL;
    constexpr std::uint64_t Rank8 = Rank1 << 56;

    const int     f     = sq & 7;
    std::uint64_t files = FileA << f;
    if (f > 0)
        files |= FileA << (f - 1);
    if (f < 7)
        files |= FileA << (f + 1);
    return files & ~Rank1 & ~Rank8 & ~(1ULL << sq);
}

using ThreatOffsetTable = std::array<std::array<int, 66>, ATTACK_TYPE_NB>;

struct ThreatFeatureCalculation {
    ThreatOffsetTable table;
    int               totalfeatures;
};

constexpr auto threatfeaturecalc = []() {
    ThreatOffsetTable t{};

    constexpr auto pseudo_attacks = bb::detail::generatePseudoAttacks();
    int            pieceoffset    = 0;

    // Count valid (non-excluded) slots for a given AttackType
    auto num_slots = [](int at) constexpr {
        int count = 0;
        for (int tt = 0; tt < TARGET_TYPE_NB; ++tt)
            if (slot_map[at][tt] >= 0)
                ++count;
        return count;
    };

    for (int at = 0; at < ATTACK_TYPE_NB; ++at)
    {
        t[at][65]     = pieceoffset;
        int squareoffset = 0;

        for (int from = (int) a1; from <= (int) h8; ++from)
        {
            t[at][from]  = squareoffset;
            bool inRange = (from >= (int) a2 && from <= (int) h7);

            if (at == W_PAWN_DIAG_AT)
            {
                if (inRange)
                    squareoffset +=
                      bb::pawnAttacks(Bitboard::square(Square(from)), Color::White).count();
            }
            else if (at == B_PAWN_DIAG_AT)
            {
                if (inRange)
                    squareoffset +=
                      bb::pawnAttacks(Bitboard::square(Square(from)), Color::Black).count();
            }
            else if (at == W_PAWN_PAIR_AT || at == B_PAWN_PAIR_AT)
            {
                if (inRange)
                    squareoffset += Bitboard::fromBits(pawn_pair_mask(from)).count();
            }
            else if (num_slots(at) > 0)
            {
                // Non-pawn valid types: at in {2,3,4,5,10,11,12,13}.
                // (at & 7) gives {2,3,4,5} for both W_ and B_ non-pawn → pt = (at&7)-1 = 1..4.
                int pt = (at & 7) - 1;  // Knight=1, Bishop=2, Rook=3, Queen=4
                squareoffset += pseudo_attacks[(PieceType) pt][Square(from)].count();
            }
            // else: gap AT (7) — num_slots=0, contributes 0; squareoffset stays 0.
        }

        t[at][64] = squareoffset;
        pieceoffset += num_slots(at) * squareoffset;
    }
    return ThreatFeatureCalculation{t, pieceoffset};
}();

constexpr ThreatOffsetTable threatoffsets  = threatfeaturecalc.table;
constexpr int               threatfeatures = threatfeaturecalc.totalfeatures;
static_assert(threatfeatures == 59808);

struct FullThreats {
    static constexpr std::string_view NAME = "Full_Threats";

    static constexpr int SQUARE_NB           = 64;
    static constexpr int COLOR_NB            = 2;
    static constexpr int MAX_ACTIVE_FEATURES = 224;

    static constexpr int INPUTS = threatfeatures;

    // clang-format off
    static constexpr Square OrientTBL[COLOR_NB][SQUARE_NB] = {
      { a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1 },
      { a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8 }
    };

    // clang-format on

    static int threat_index(Color      perspective,
                            AttackType at,
                            Square     from,
                            Square     to,
                            TargetType tt,
                            Square     ksq) {
        int    orient       = (int) OrientTBL[(int) perspective][(int) ksq];
        Square from_oriented = (Square) ((int) from ^ orient);
        Square to_oriented   = (Square) ((int) to   ^ orient);

        // Flip W<->B when computing from Black's perspective.
        // XOR-spacing-8: W_=0-5, B_=8-13 for AT; W_=0-4, B_=8-12 for TT.
        // XOR with 8 swaps between the two ranges without branches.
        int pxor       = (perspective == Color::Black) ? 8 : 0;
        AttackType at_oriented = AttackType(at ^ pxor);
        TargetType tt_oriented = TargetType(tt ^ pxor);

        // Full exclusion
        int8_t slot = slot_map[at_oriented][tt_oriented];
        if (slot < 0)
            return -1;

        // Semi-exclusion: FROM_GT convention — exclude when from_oriented < to_oriented
        if (semi_map[at_oriented][tt_oriented] && from_oriented < to_oriented)
            return -1;

        // Pseudo-attacks from from_oriented under at_oriented (used for popcount rank)
        Bitboard attacks;
        if (at_oriented == W_PAWN_DIAG_AT)
            attacks = bb::pawnAttacks(Bitboard::square(from_oriented), Color::White);
        else if (at_oriented == B_PAWN_DIAG_AT)
            attacks = bb::pawnAttacks(Bitboard::square(from_oriented), Color::Black);
        else if (at_oriented == W_PAWN_PAIR_AT || at_oriented == B_PAWN_PAIR_AT)
            attacks = Bitboard::fromBits(pawn_pair_mask((int) from_oriented));
        else
        {
            // Non-pawn: at_oriented in {2,3,4,5} (W_) or {10,11,12,13} (B_).
            // (at_oriented & 7) gives {2,3,4,5} for both → pt = (at_oriented & 7) - 1 = 1..4.
            int pt = (at_oriented & 7) - 1;  // Knight=1, Bishop=2, Rook=3, Queen=4
            attacks = bb::detail::pseudoAttacks()[(PieceType) pt][from_oriented];
        }
        return int(threatoffsets[at_oriented][65]
                   + slot * threatoffsets[at_oriented][64]
                   + threatoffsets[at_oriented][(int) from_oriented]
                   + (Bitboard::fromBits((1ULL << (int) to_oriented) - 1) & attacks).count());
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos       = e.pos;
        auto  pieces    = pos.piecesBB();
        // Kings are never targets; exclude them upfront (mirrors Stockfish's occupiedNoK)
        auto  piecesNoK = pieces & ~pos.piecesBB(whiteKing) & ~pos.piecesBB(blackKing);
        auto  ksq       = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k         = 0;

        for (int i = (int) Color::White; i <= (int) Color::Black; i++)
        {
            Color c = order[(int) color][i];

            // ---- Pawn attacks ----
            {
                Piece      attkr    = Piece(PieceType::Pawn, c);
                Bitboard   bb       = pos.piecesBB(attkr);
                AttackType at_diag  = make_attack_type(attkr);
                auto       allPawns = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);

                // Diagonal pawn threats exclude pawn targets (pawn-on-pawn co-presence
                // is captured by PP_3Wide).
                auto diagTargets = piecesNoK & ~allPawns;

                auto right    = (c == Color::White) ? Offset(1, 1)  : Offset(-1, -1);
                auto left     = (c == Color::White) ? Offset(-1, 1) : Offset(1, -1);
                int  r_delta  = (c == Color::White) ? 9  : -9;
                int  l_delta  = (c == Color::White) ? 7  : -7;

                for (Square to : bb.shifted(right) & diagTargets)
                {
                    Square     from  = Square((int) to - r_delta);
                    TargetType tt    = make_target_type(pos.pieceAt(to));
                    int        index = threat_index(color, at_diag, from, to, tt, ksq);
                    if (index >= 0) { features[k++] = index; }
                }
                for (Square to : bb.shifted(left) & diagTargets)
                {
                    Square     from  = Square((int) to - l_delta);
                    TargetType tt    = make_target_type(pos.pieceAt(to));
                    int        index = threat_index(color, at_diag, from, to, tt, ksq);
                    if (index >= 0) { features[k++] = index; }
                }
            }

            // ---- Non-pawn attacks (Knight through Queen; Kings are never attackers here) ----
            for (int j = (int) PieceType::Knight; j < (int) PieceType::King; j++)
            {
                Piece      attkr = Piece(PieceType(j), c);
                AttackType at    = make_attack_type(attkr);
                Bitboard   bb    = pos.piecesBB(attkr);

                for (Square from : bb)
                {
                    Bitboard attacks = pos.attacks(from) & piecesNoK;
                    for (Square to : attacks)
                    {
                        TargetType tt    = make_target_type(pos.pieceAt(to));
                        int        index = threat_index(color, at, from, to, tt, ksq);
                        if (index >= 0) { features[k++] = index; }
                    }
                }
            }
        }
        return {k, INPUTS};
    }
};

struct FullThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return FullThreats::INPUTS; }
    int max_active_features() const override { return FullThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             Color                    color) const override {
        return FullThreats::fill_features_sparse(e, features, color);
    }
};

struct PP_3Wide {
    static constexpr std::string_view NAME = "PP_3Wide";

    static constexpr int PAWN_IDS            = 2 * 48;
    static constexpr int INPUTS              = PAWN_IDS * (PAWN_IDS - 1) / 2;
    static constexpr int MAX_ACTIVE_FEATURES = 128;

    static int make_pawn_id(Color color, Square square) {
        return 48 * static_cast<int>(color) + static_cast<int>(square) - static_cast<int>(a2);
    }

    static int make_index(Color perspective,
                          Color color,
                          Square from,
                          Square to,
                          Color paired_color,
                          Square ksq) {
        int orient = static_cast<int>(FullThreats::OrientTBL[static_cast<int>(perspective)]
                                                             [static_cast<int>(ksq)]);
        Square from_oriented = static_cast<Square>(static_cast<int>(from) ^ orient);
        Square to_oriented   = static_cast<Square>(static_cast<int>(to) ^ orient);

        Color color_oriented =
          static_cast<Color>(static_cast<int>(color) ^ static_cast<int>(perspective));
        Color paired_color_oriented =
          static_cast<Color>(static_cast<int>(paired_color) ^ static_cast<int>(perspective));

        if (from_oriented < a2 || from_oriented > h7 || to_oriented < a2 || to_oriented > h7)
            return INPUTS;

        int id_a = make_pawn_id(color_oriented, from_oriented);
        int id_b = make_pawn_id(paired_color_oriented, to_oriented);
        int hi   = std::max(id_a, id_b);
        int lo   = std::min(id_a, id_b);

        return hi * (hi - 1) / 2 + lo;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos      = e.pos;
        auto  allPawns = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        auto  ksq      = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k        = 0;

        for (int i = static_cast<int>(Color::White); i <= static_cast<int>(Color::Black); ++i)
        {
            Color c  = order[static_cast<int>(color)][i];
            Piece p  = Piece(PieceType::Pawn, c);
            Bitboard bb = pos.piecesBB(p);

            for (Square from : bb)
            {
                Bitboard targets = Bitboard::fromBits(pawn_pair_mask(static_cast<int>(from))) & allPawns;
                for (Square to : targets)
                {
                    if (from < to)
                        continue;

                    Color paired_color = pos.pieceAt(to).color();
                    int   index        = make_index(color, c, from, to, paired_color, ksq);
                    if (index < INPUTS)
                    {
                        features[k] = index;
                        ++k;
                    }
                }
            }
        }

        return {k, INPUTS};
    }
};

struct PP_3WideExtractor: IFeatureExtractor {
    int inputs() const override { return PP_3Wide::INPUTS; }
    int max_active_features() const override { return PP_3Wide::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             Color                    color) const override {
        return PP_3Wide::fill_features_sparse(e, features, color);
    }
};

struct ComposedFeatureExtractor: IFeatureExtractor {
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors;
    int                                             m_inputs;
    int                                             m_max_active;

    ComposedFeatureExtractor(std::vector<std::unique_ptr<IFeatureExtractor>> exts) :
        extractors(std::move(exts)),
        m_inputs(0),
        m_max_active(0) {
        for (auto& e : extractors)
        {
            m_inputs += e->inputs();
            m_max_active += e->max_active_features();
        }
    }

    int inputs() const override { return m_inputs; }
    int max_active_features() const override { return m_max_active; }

    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             Color                    color) const override {
        int total_written = 0;
        int input_offset  = 0;

        for (auto& ext : extractors)
        {
            auto [written, ext_inputs] =
              ext->fill_features_sparse(e, features + total_written, color);

            // Offset the feature indices for this component
            for (int i = 0; i < written; ++i)
                features[total_written + i] += input_offset;

            input_offset += ext_inputs;
            total_written += written;
        }

        return {total_written, m_inputs};
    }
};

static std::unique_ptr<IFeatureExtractor> make_single_extractor(std::string_view name) {
    if (name == "HalfKAv2_hm")
        return std::make_unique<HalfKAv2_hmExtractor>();
    if (name == "Full_Threats")
        return std::make_unique<FullThreatsExtractor>();
    if (name == "PP_3Wide")
        return std::make_unique<PP_3WideExtractor>();
    return nullptr;
}

std::shared_ptr<IFeatureExtractor> get_feature(std::string_view name) {
    std::vector<std::unique_ptr<IFeatureExtractor>> components;
    std::size_t                                     start = 0;

    while (start < name.size())
    {
        auto pos  = name.find('+', start);
        auto part = name.substr(start, pos == std::string_view::npos ? pos : pos - start);
        auto ext  = make_single_extractor(part);

        if (!ext)
        {
            std::cerr << "Unknown feature component: " << part << std::endl;
            return nullptr;
        }

        components.push_back(std::move(ext));
        start = (pos == std::string_view::npos) ? name.size() : pos + 1;
    }

    if (components.empty())
        return nullptr;

    if (components.size() == 1)
        return std::shared_ptr<IFeatureExtractor>(std::move(components[0]));

    return std::make_shared<ComposedFeatureExtractor>(std::move(components));
}

// ---------------------------------------------------------
// Class Implementations
// ---------------------------------------------------------

template <typename T>
struct BumpAllocator {
    T* ptr;
    BumpAllocator(T* block) : ptr(block) {}
    T* alloc(size_t count) {
        T* res = ptr;
        ptr += count;
        return res;
    }
};

SparseBatch::SparseBatch(const IFeatureExtractor&              feature_set,
                         const std::vector<TrainingDataEntry>& entries)
#ifdef NNUE_LOADER_STATISTICS
    :
    entries_copy(entries)
#endif
{
    num_inputs          = feature_set.inputs();
    size                = entries.size();
    max_active_features = feature_set.max_active_features();
    const size_t total_floats = size * 3;
    const size_t total_ints   = size + size * max_active_features * 2;

    m_float_block = new float[total_floats];
    m_int_block   = new int[total_ints];

    BumpAllocator<float> float_alloc(m_float_block);
    is_white     = float_alloc.alloc(size);
    outcome      = float_alloc.alloc(size);
    score        = float_alloc.alloc(size);

    BumpAllocator<int> int_alloc(m_int_block);
    white               = int_alloc.alloc(size * max_active_features);
    black               = int_alloc.alloc(size * max_active_features);
    piece_count         = int_alloc.alloc(size);

    num_active_white_features = 0;
    num_active_black_features = 0;

    for (int i = 0; i < size * max_active_features; ++i)
        white[i] = -1;
    for (int i = 0; i < size * max_active_features; ++i)
        black[i] = -1;

    for (int i = 0; i < size; ++i)
        fill_entry(feature_set, i, entries[i]);
}

SparseBatch::~SparseBatch() {
    delete[] m_float_block;
    delete[] m_int_block;
}

void SparseBatch::fill_entry(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e) {
    is_white[i]            = static_cast<float>(e.pos.sideToMove() == Color::White);
    outcome[i]             = (e.result + 1.0f) / 2.0f;
    score[i]               = e.score;
    piece_count[i]         = e.pos.piecesBB().count();
    fill_features(fs, i, e);
}

void SparseBatch::fill_features(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e) {
    const int offset = i * max_active_features;
    num_active_white_features +=
      fs.fill_features_sparse(e, white + offset, Color::White).first;
    num_active_black_features +=
      fs.fill_features_sparse(e, black + offset, Color::Black).first;
}

int FeaturedBatchStream::calculate_num_reader_threads(int concurrency) {
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FeaturedBatchStream::calculate_num_worker_threads(int concurrency) {
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FeaturedBatchStream::FeaturedBatchStream(
  std::shared_ptr<IFeatureExtractor>            feature_set,
  int                                           concurrency,
  const std::vector<std::string>&               filenames,
  int                                           batch_size,
  bool                                          cyclic,
  std::function<bool(const TrainingDataEntry&)> skipPredicate,
  int                                           rank,
  int                                           world_size) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             rank,
             world_size),
    m_feature_set(std::move(feature_set)),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency)) {

    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty())
                    break;
            }

            auto batch = new SparseBatch(*m_feature_set, entries);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < static_cast<size_t>(m_concurrency) + 1 || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i)
    {
        m_workers.emplace_back(worker);
    }
}

FeaturedBatchStream::~FeaturedBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
}

SparseBatch* FeaturedBatchStream::next() {
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty())
    {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

Fen::Fen() :
    m_fen(nullptr) { }

Fen::Fen(const std::string& fen) :
    m_size(fen.size()),
    m_fen(new char[fen.size() + 1]) {
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
}

Fen& Fen::operator=(const std::string& fen) {
    if (m_fen != nullptr)
        delete[] m_fen;
    m_size = fen.size();
    m_fen  = new char[fen.size() + 1];
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    return *this;
}

Fen::~Fen() { delete[] m_fen; }

FenBatch::FenBatch(const std::vector<TrainingDataEntry>& entries) :
    m_size(entries.size()),
    m_fens(new Fen[entries.size()]) {
    for (int i = 0; i < m_size; ++i)
        m_fens[i] = entries[i].pos.fen();
}

FenBatch::~FenBatch() { delete[] m_fens; }

int FenBatchStream::calculate_num_reader_threads(int concurrency) {
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FenBatchStream::calculate_num_worker_threads(int concurrency) {
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FenBatchStream::FenBatchStream(int                                           concurrency,
                               const std::vector<std::string>&               filenames,
                               int                                           batch_size,
                               bool                                          cyclic,
                               std::function<bool(const TrainingDataEntry&)> skipPredicate,
                               int                                           rank,
                               int                                           world_size) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             rank,
             world_size),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency)) {

    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty())
                    break;
            }

            auto batch = new FenBatch(entries);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < static_cast<size_t>(m_concurrency) + 1 || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i)
    {
        m_workers.emplace_back(worker);
    }
}

FenBatchStream::~FenBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
}

FenBatch* FenBatchStream::next() {
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty())
    {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config) {
    if (!config.filtered && !config.wld_filtered && config.random_fen_skipping <= 0
        && config.early_fen_skipping < 0 && config.soft_early_fen_skipping <= 0)
    {
        return nullptr;
    }

    double   skip_prob             = 0.0;
    uint64_t random_skip_threshold = 0;
    if (config.random_fen_skipping > 0)
    {
        skip_prob = double(config.random_fen_skipping) / (config.random_fen_skipping + 1);
        random_skip_threshold = static_cast<uint64_t>(skip_prob * static_cast<double>(~0ULL));
    }

    // --- Precompute 5-Point Spline PC LUT ---
    std::array<double, 33> target_pc_weights_lut{};
    double                 target_pc_weights_total = 0.0;

    auto desired_piece_count_weights = [&config](int pc) -> double {
        double x    = static_cast<double>(pc);
        double y[5] = {config.pc_y0, config.pc_y1, config.pc_y2, config.pc_y3, config.pc_y4};

        if (x <= 0)
            return y[0];
        if (x >= 32)
            return y[4];

        int i = static_cast<int>(x / 8.0);
        if (i > 3)
            i = 3;

        double x0 = i * 8.0;
        double t  = (x - x0) / 8.0;

        auto get_slope = [&](int idx) {
            if (idx == 0)
                return (y[1] - y[0]);
            if (idx == 4)
                return (y[4] - y[3]);
            return (y[idx + 1] - y[idx - 1]) / 2.0;
        };

        double m0 = get_slope(i);
        double m1 = get_slope(i + 1);

        double t2 = t * t;
        double t3 = t2 * t;

        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;

        double val = h00 * y[i] + h10 * m0 + h01 * y[i + 1] + h11 * m1;

        return std::max(0.0, val);
    };

    for (int i = 0; i < 33; ++i)
    {
        target_pc_weights_lut[i] = desired_piece_count_weights(i);
        target_pc_weights_total += target_pc_weights_lut[i];
    }
    if (target_pc_weights_total <= 0.0)
    {
        std::fill(target_pc_weights_lut.begin(), target_pc_weights_lut.end(), 1.0);
        target_pc_weights_total = static_cast<double>(target_pc_weights_lut.size());
    }

    // --- Precompute Soft Early Ply Filter LUT ---
    std::vector<double> early_ply_accept_prob;

    if (config.soft_early_fen_skipping > 0)
    {
        size_t lut_size = static_cast<size_t>(config.soft_early_fen_skipping) + 1;
        early_ply_accept_prob.resize(lut_size);

        auto interpolate_ply = [&config](double ply) -> double {
            struct Pt {
                double x, y;
            };
            Pt pts[5] = {{config.ply_x1, config.ply_y1},
                         {config.ply_x2, config.ply_y2},
                         {config.ply_x3, config.ply_y3},
                         {config.ply_x4, config.ply_y4},
                         {static_cast<double>(config.soft_early_fen_skipping), 1.0}};

            if (ply <= pts[0].x)
                return pts[0].y;
            if (ply >= pts[4].x)
                return pts[4].y;

            for (int i = 0; i < 4; ++i)
            {
                if (ply >= pts[i].x && ply <= pts[i + 1].x)
                {
                    if (pts[i + 1].x == pts[i].x)
                        return pts[i].y;
                    double t = (ply - pts[i].x) / (pts[i + 1].x - pts[i].x);
                    return pts[i].y + t * (pts[i + 1].y - pts[i].y);
                }
            }
            return 1.0;
        };

        for (size_t i = 0; i < lut_size; ++i)
        {
            early_ply_accept_prob[i] =
              std::clamp(interpolate_ply(static_cast<double>(i)), 0.0, 1.0);
        }
    }

    return [config, random_skip_threshold, target_pc_weights_lut, target_pc_weights_total,
            early_ply_accept_prob = std::move(early_ply_accept_prob)](const TrainingDataEntry& e) {
        static constexpr int    VALUE_NONE = 32002;
        static thread_local int last_ply   = -1;
        static thread_local int last_score = VALUE_NONE;

        // skip when score 0 was used as placeholder
        // detected through heuristic:
        // Game was not a draw and
        // last valid score was somewhat large

        bool skip_placeholder_zero = e.ply > last_ply && last_score != VALUE_NONE
                                  && std::abs(last_score) > 100 && e.result != 0 && e.score == 0;

        last_ply = e.ply;

        if (e.score == VALUE_NONE)
            return true;
        if (skip_placeholder_zero)
            return true;

        // Only update if valid score.
        last_score = e.score;

        // Hard Early Ply Filter
        if (e.ply <= config.early_fen_skipping)
            return true;

        auto& prng = rng::get_thread_local_rng();

        if (config.random_fen_skipping && (prng() < random_skip_threshold))
            return true;
        if (config.filtered && (e.isCapturingMove() || e.isInCheck()))
            return true;

        if (config.wld_filtered)
        {
            uint64_t wld_skip_threshold =
              static_cast<uint64_t>((1.0 - e.score_result_prob()) * static_cast<double>(~0ULL));
            if (prng() < wld_skip_threshold)
                return true;
        }

        if (config.simple_eval_skipping > 0
            && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping)
        {
            return true;
        }

        // Soft Early Ply Filter
        if (config.soft_early_fen_skipping > 0 && e.ply < config.soft_early_fen_skipping)
        {
            uint64_t ply_reject_threshold = static_cast<uint64_t>(
              (1.0 - early_ply_accept_prob[e.ply]) * static_cast<double>(~0ULL));
            if (prng() < ply_reject_threshold)
            {
                return true;
            }
        }

        // Dynamic Piece Count Filter
        const int pc = e.pos.piecesBB().count();
        if (pc < 0 || pc > 32)
            return true;

        static thread_local double   alpha                   = 1.0;
        static thread_local double   pc_history_all[33]      = {0};
        static thread_local double   pc_history_passed[33]   = {0};
        static thread_local double   pc_history_all_total    = 0;
        static thread_local double   pc_history_passed_total = 0;
        static thread_local uint64_t step_count              = 0;

        const double max_pc_skip_rate = 0.975;

        pc_history_all[pc] += 1.0;
        pc_history_all_total += 1.0;
        step_count++;

        bool should_update =
          (step_count == 100 || step_count == 500 || step_count == 1000 || step_count == 2500
           || step_count == 5000 || (step_count > 5000 && step_count % 10000 == 0));

        if (should_update)
        {
            double min_ratio   = std::numeric_limits<double>::infinity();
            bool   found_valid = false;

            for (int i = 0; i < 33; ++i)
            {
                if (target_pc_weights_lut[i] > 0.0 && pc_history_all[i] > 0.0)
                {
                    double current_ratio = (pc_history_all_total * target_pc_weights_lut[i])
                                         / (target_pc_weights_total * pc_history_all[i]);
                    if (current_ratio < min_ratio)
                    {
                        min_ratio   = current_ratio;
                        found_valid = true;
                    }
                }
            }
            if (found_valid && min_ratio > 0.0)
            {
                alpha = (1.0 - max_pc_skip_rate) / min_ratio;
            }

            if (step_count >= 10000 && step_count % 10000 == 0)
            {
                for (int i = 0; i < 33; ++i)
                {
                    pc_history_all[i] *= 0.5;
                }
                pc_history_all_total *= 0.5;
            }
        }

        double accept_prob = 0.0;
        if (target_pc_weights_lut[pc] > 0.0 && pc_history_all[pc] > 0.0)
        {
            double current_ratio = (pc_history_all_total * target_pc_weights_lut[pc])
                                 / (target_pc_weights_total * pc_history_all[pc]);
            accept_prob = alpha * current_ratio;
        }

        accept_prob = std::clamp(accept_prob, 0.0, 1.0);

        uint64_t reject_threshold =
          static_cast<uint64_t>((1.0 - accept_prob) * static_cast<double>(~0ULL));
        if (prng() < reject_threshold)
        {
            return true;
        }

        pc_history_passed[pc] += 1.0;
        pc_history_passed_total += 1.0;

        return false;
    };
}
