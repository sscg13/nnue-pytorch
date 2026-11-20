from .k32h3 import K32H3, KingBuckets


def _merge_king_bucket(bucket: int) -> int:
    if bucket < 16:
        return 2 * (bucket // 8) + (bucket % 4) // 2
    return bucket - 12


class K20H3(K32H3):
    HASH = 0x56FEAFF6
    FEATURE_NAME = "K20H3^"
    INPUT_FEATURE_NAME = "K20H3"

    NUM_KING_BUCKETS = 20
    NUM_BUCKETS = NUM_KING_BUCKETS * K32H3.NUM_STATES
    NUM_INPUTS = K32H3.NUM_PLANES * NUM_BUCKETS
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS

    # Every pair of squares in one 2x2 region is adjacent. The opposing king
    # cannot legally use any of these squares while the own king is in the bucket.
    KING_SQUARES = tuple(
        tuple(
            sq
            for sq, bucket in enumerate(KingBuckets)
            if bucket >= 0 and _merge_king_bucket(bucket) == group
        )
        for group in range(NUM_KING_BUCKETS)
    )
