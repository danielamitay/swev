"""Small exporter checks that do not load checkpoint weights."""
import unittest

from convert_gemma import LABELS, attention_masks, sequence_buckets
import torch


class GemmaExportTests(unittest.TestCase):
    def test_sliding_attention_excludes_distant_keys_only(self):
        length = 1024
        positions = torch.arange(length)
        full = torch.where(positions[None, :] <= positions[:, None], 0.0, -10000.0)[None, None]
        masks = attention_masks(torch.zeros((1, length), dtype=torch.int32), full, 512)
        self.assertTrue(torch.equal(masks["full_attention"], full))
        expected = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - 512)
        self.assertTrue(torch.equal(masks["sliding_attention"] == 0, expected[None, None]))
        self.assertEqual(masks["sliding_attention"][0, 0, 512, 0], -10000)
        self.assertEqual(masks["sliding_attention"][0, 0, 512, 1], 0)

    def test_capacity_and_buckets(self):
        self.assertEqual(LABELS, list("ABCDEFGHIJKLMNOP"))
        self.assertEqual(sequence_buckets(4096), [128, 256, 512, 1024, 2048, 4096])
        self.assertEqual(sequence_buckets(300), [128, 256, 300])
        self.assertEqual(sequence_buckets(64), [64])


if __name__ == "__main__":
    unittest.main()
