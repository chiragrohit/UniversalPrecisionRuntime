import unittest
import torch
import numpy as np
from upr.bit_ops import (
    float16_to_uint16_numpy,
    uint16_to_float16_torch,
    extract_bit_plane_np,
    pack_bit_plane,
    unpack_bit_plane,
    reconstruct_tensor
)
from upr.metrics import compute_weight_metrics

class TestBitOps(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_float16_uint16_roundtrip(self):
        orig_tensor = torch.randn(10, 10, dtype=torch.float16)
        uint16_arr = float16_to_uint16_numpy(orig_tensor)
        recon_tensor = uint16_to_float16_torch(uint16_arr)
        self.assertTrue(torch.equal(orig_tensor, recon_tensor))

    def test_bit_packing_unpacking(self):
        shape = (128, 64)
        bit_arr = np.random.randint(0, 2, size=shape, dtype=np.uint8)
        packed_bytes = pack_bit_plane(bit_arr)
        unpacked_arr = unpack_bit_plane(packed_bytes, num_elements=128 * 64, shape=shape)
        np.testing.assert_array_equal(bit_arr, unpacked_arr)

    def test_16bit_lossless_reconstruction(self):
        shape = (32, 32)
        orig_tensor = torch.randn(*shape, dtype=torch.float16)
        uint16_arr = float16_to_uint16_numpy(orig_tensor)

        planes_dict = {}
        for b in range(16):
            bit_arr = extract_bit_plane_np(uint16_arr, b)
            packed = pack_bit_plane(bit_arr)
            planes_dict[b] = packed

        recon_tensor = reconstruct_tensor(
            planes_dict=planes_dict,
            selected_bits=16,
            original_shape=shape
        )

        metrics = compute_weight_metrics(orig_tensor, recon_tensor)
        self.assertTrue(metrics["exact_match"])
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["cosine_similarity"], 1.0)

    def test_variable_bits_reconstruction(self):
        shape = (16, 16)
        orig_tensor = torch.randn(*shape, dtype=torch.float16)
        uint16_arr = float16_to_uint16_numpy(orig_tensor)

        planes_dict = {}
        for b in range(16):
            bit_arr = extract_bit_plane_np(uint16_arr, b)
            packed = pack_bit_plane(bit_arr)
            planes_dict[b] = packed

        # Test 8 bits (MSBs 15..8 present, LSBs 7..0 zero-filled)
        recon_8bit = reconstruct_tensor(
            planes_dict=planes_dict,
            selected_bits=8,
            original_shape=shape
        )
        metrics_8bit = compute_weight_metrics(orig_tensor, recon_8bit)
        self.assertFalse(metrics_8bit["exact_match"])
        self.assertGreater(metrics_8bit["cosine_similarity"], 0.95)

if __name__ == "__main__":
    unittest.main()
