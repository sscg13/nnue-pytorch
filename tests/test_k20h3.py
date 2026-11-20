import unittest

import torch

from model.modules.features import K20H3, get_feature_cls


class TestK20H3(unittest.TestCase):
    def test_bucket_geometry(self):
        self.assertIs(get_feature_cls("Full_Threats+PP_3Wide+K20H3^")[-1], K20H3)
        groups = [K20H3.own_king_squares(bucket) for bucket in range(20)]
        self.assertEqual(
            groups[:4],
            [(54, 55, 62, 63), (52, 53, 60, 61), (38, 39, 46, 47), (36, 37, 44, 45)],
        )
        self.assertEqual(groups[4:], [(sq,) for sq in range(31, -1, -1) if sq % 8 >= 4])
        self.assertEqual(K20H3.NUM_INPUTS, 46_080)
        self.assertEqual(K20H3.NUM_REAL_FEATURES, 42_240)

    @torch.no_grad()
    def test_merge_and_expand_king_planes(self):
        feature = K20H3(1)
        feature.weight[:, 0] = torch.arange(K20H3.NUM_INPUTS)
        feature.virtual_weight[:, 0] = torch.arange(K20H3.NUM_PLANES) / 4
        merged = feature.merged_weight().reshape(K20H3.NUM_BUCKETS, 12, 64)
        exported = feature.get_export_weights()
        planes = exported.reshape(K20H3.NUM_BUCKETS, 11, 64)
        torch.testing.assert_close(planes[:, :10], merged[:, :10], rtol=0, atol=0)

        feature.load_export_weights(exported)
        loaded = feature.weight.reshape(K20H3.NUM_BUCKETS, 12, 64)
        self.assertEqual(feature.virtual_weight.count_nonzero().item(), 0)
        for bucket in range(K20H3.NUM_BUCKETS):
            own_squares = list(K20H3.own_king_squares(bucket))
            other_squares = [sq for sq in range(64) if sq not in own_squares]
            self.assertEqual(loaded[bucket, 11, own_squares].count_nonzero().item(), 0)
            self.assertEqual(loaded[bucket, 10, other_squares].count_nonzero().item(), 0)
            torch.testing.assert_close(
                loaded[bucket, 10, own_squares], merged[bucket, 10, own_squares], rtol=0, atol=0
            )
            torch.testing.assert_close(
                loaded[bucket, 11, other_squares], merged[bucket, 11, other_squares], rtol=0, atol=0
            )
        torch.testing.assert_close(feature.get_export_weights(), exported, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
