import numpy as np
import pytest
import math
from upr.bit_ops import pack_bit_plane, unpack_bit_plane, get_packing_stats

def test_bit_packing_unpacking_roundtrip():
    np.random.seed(42)
    bits = np.random.randint(0, 2, size=(10, 10), dtype=np.uint8)

    packed = pack_bit_plane(bits)
    expected_bytes = math.ceil(100 / 8)
    assert len(packed) == expected_bytes

    unpacked = unpack_bit_plane(packed, num_elements=100, shape=(10, 10))
    assert np.array_equal(bits, unpacked)

def test_packing_stats():
    stats = get_packing_stats(num_elements=1000, bits_reconstructed=8)
    assert stats["raw_fp16_bytes"] == 2000
    assert stats["packed_bits_bytes"] == 1000
    assert stats["compression_ratio"] == 2.0
