"""
Simple test script to verify compression functionality.
"""

import numpy as np
import os
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import my_utils.array_logger as alog


def test_compression():
    """Test array_logger with compression enabled."""
    print("=" * 60)
    print("Testing array_logger with compression")
    print("=" * 60)
    
    # Clean up old test data
    test_dir = "./test_logs_compressed"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Test 1: Write with compression
    print("\n[Test 1] Writing data with compression (level 3)...")
    alog.init(test_dir, compression_level=3)
    alog.register("test_data", keys=["episode", "step"], shape=(10, 5), dtype="float32")
    
    for episode in range(5):
        for step in range(10):
            data = np.random.rand(10, 5).astype(np.float32)
            alog.log("test_data", {"episode": episode, "step": step}, data)
    
    alog.close()
    print("✓ Data written successfully")
    
    # Check that .db.zst file exists
    compressed_file = f"{test_dir}/test_data.db.zst"
    if os.path.exists(compressed_file):
        size_mb = os.path.getsize(compressed_file) / (1024 * 1024)
        print(f"✓ Compressed file created: {compressed_file} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Compressed file not found: {compressed_file}")
        return False
    
    # Check that .db file does not exist
    db_file = f"{test_dir}/test_data.db"
    if not os.path.exists(db_file):
        print("✓ Original .db file removed")
    else:
        print(f"✗ Original .db file still exists: {db_file}")
        return False
    
    # Test 2: Read compressed data
    print("\n[Test 2] Reading compressed data...")
    reader = alog.open_reader(f"{test_dir}/test_data.db")
    
    count = 0
    for keys_dict, array in reader.iterate():
        count += 1
        assert array.shape == (10, 5), f"Shape mismatch: {array.shape}"
        assert array.dtype == np.float32, f"Dtype mismatch: {array.dtype}"
    
    reader.close()
    print(f"✓ Successfully read {count} records")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("\n✓ All tests passed!")
    return True


def test_no_compression():
    """Test array_logger without compression."""
    print("\n" + "=" * 60)
    print("Testing array_logger without compression")
    print("=" * 60)
    
    # Clean up old test data
    test_dir = "./test_logs_uncompressed"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Test: Write without compression
    print("\n[Test] Writing data without compression...")
    alog.init(test_dir, compression_level=None)
    alog.register("test_data", keys=["episode"], shape=(5,), dtype="float32")
    
    for episode in range(10):
        data = np.random.rand(5).astype(np.float32)
        alog.log("test_data", {"episode": episode}, data)
    
    alog.close()
    print("✓ Data written successfully")
    
    # Check that .db file exists
    db_file = f"{test_dir}/test_data.db"
    if os.path.exists(db_file):
        size_kb = os.path.getsize(db_file) / 1024
        print(f"✓ Uncompressed file created: {db_file} ({size_kb:.2f} KB)")
    else:
        print(f"✗ DB file not found: {db_file}")
        return False
    
    # Check that .db.zst file does not exist
    compressed_file = f"{test_dir}/test_data.db.zst"
    if not os.path.exists(compressed_file):
        print("✓ No compressed file created")
    else:
        print(f"✗ Unexpected compressed file: {compressed_file}")
        return False
    
    # Read uncompressed data
    print("\nReading uncompressed data...")
    reader = alog.open_reader(f"{test_dir}/test_data.db")
    
    count = 0
    for keys_dict, array in reader.iterate():
        count += 1
    
    reader.close()
    print(f"✓ Successfully read {count} records")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success1 = test_compression()
        success2 = test_no_compression()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED ✓")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED ✗")
            print("=" * 60)
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
