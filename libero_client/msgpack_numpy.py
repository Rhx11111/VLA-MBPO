"""
MessagePack serialization support for NumPy arrays.
Based on msgpack-numpy but simplified for our use case.
"""

import numpy as np
import msgpack


def encode_numpy(obj):
    """
    Encode numpy arrays for msgpack serialization.
    
    Args:
        obj: Object to encode (numpy array or other)
        
    Returns:
        Encoded object or original object if not numpy array
    """
    if isinstance(obj, np.ndarray):
        return {
            '__numpy__': True,
            'dtype': str(obj.dtype),
            'shape': obj.shape,
            'data': obj.tobytes()
        }
    return obj


def decode_numpy(obj):
    """
    Decode numpy arrays from msgpack serialization.
    
    Args:
        obj: Object to decode
        
    Returns:
        Decoded numpy array or original object
    """
    if isinstance(obj, dict) and obj.get('__numpy__'):
        return np.frombuffer(
            obj['data'], 
            dtype=obj['dtype']
        ).reshape(obj['shape'])
    return obj


def packb(obj, **kwargs):
    """
    Pack object to msgpack bytes with numpy support.
    
    Args:
        obj: Object to pack
        **kwargs: Additional arguments for msgpack.packb
        
    Returns:
        Packed bytes
    """
    return msgpack.packb(obj, default=encode_numpy, **kwargs)


def unpackb(data, **kwargs):
    """
    Unpack msgpack bytes with numpy support.
    
    Args:
        data: Bytes to unpack
        **kwargs: Additional arguments for msgpack.unpackb
        
    Returns:
        Unpacked object
    """
    return msgpack.unpackb(data, object_hook=decode_numpy, **kwargs)


class Packer:
    """
    MessagePack packer with numpy support.
    """
    
    def __init__(self, **kwargs):
        """Initialize packer with numpy support."""
        self._packer = msgpack.Packer(default=encode_numpy, **kwargs)
    
    def pack(self, obj):
        """Pack object to bytes."""
        return self._packer.pack(obj)


class Unpacker:
    """
    MessagePack unpacker with numpy support.
    """
    
    def __init__(self, **kwargs):
        """Initialize unpacker with numpy support."""
        self._unpacker = msgpack.Unpacker(object_hook=decode_numpy, **kwargs)
    
    def feed(self, data):
        """Feed data to unpacker."""
        return self._unpacker.feed(data)
    
    def unpack(self):
        """Unpack next object."""
        return self._unpacker.unpack()
    
    def __iter__(self):
        """Iterate over unpacked objects."""
        return iter(self._unpacker)


# Test function
def test_numpy_serialization():
    """Test numpy array serialization and deserialization."""
    # Test various numpy array types
    test_arrays = [
        np.array([1, 2, 3, 4, 5]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        np.random.randn(10, 20, 3),
        np.array(['hello', 'world'], dtype='U10'),
        np.array([True, False, True]),
    ]
    
    for i, arr in enumerate(test_arrays):
        print(f"Testing array {i}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Pack and unpack
        packed = packb(arr)
        unpacked = unpackb(packed)
        
        # Check equality
        if arr.dtype.kind == 'U':  # Unicode strings
            assert np.array_equal(arr, unpacked), f"Array {i} failed equality test"
        else:
            assert np.allclose(arr, unpacked), f"Array {i} failed equality test"
        
        print(f"  ✓ Passed")
    
    # Test complex object with numpy arrays
    complex_obj = {
        'image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        'action': np.random.randn(7),
        'metadata': {
            'timestamp': 123456,
            'qpos': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        },
        'text': 'hello world'
    }
    
    packed = packb(complex_obj)
    unpacked = unpackb(packed)
    
    assert np.array_equal(complex_obj['image'], unpacked['image'])
    assert np.allclose(complex_obj['action'], unpacked['action'])
    assert np.allclose(complex_obj['metadata']['qpos'], unpacked['metadata']['qpos'])
    assert complex_obj['metadata']['timestamp'] == unpacked['metadata']['timestamp']
    assert complex_obj['text'] == unpacked['text']
    
    print("✓ Complex object test passed")
    print("All tests passed!")


if __name__ == "__main__":
    test_numpy_serialization()
