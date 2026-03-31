from THREEFISH.threefish_naive import encrypt as naive_enc, decrypt as naive_dec
from THREEFISH.threefish_optimized import encrypt as opt_enc, decrypt as opt_dec
from THREEFISH.threefish_numba import encrypt as numba_enc, decrypt as numba_dec
from THREEFISH.threefish_gpu import encrypt as gpu_enc, decrypt as gpu_dec


def test_threefish_naive():
    key = bytes(range(32))
    data = b"This is a test message for Threefish cipher!"

    ct = naive_enc(data, key)
    pt = naive_dec(ct, key)

    print("\n[NAIVE]")
    print("Ciphertext:", ct.hex())
    print("Plain text:", pt)

    assert pt == data
    print("Naive test passed!")


def test_threefish_optimized():
    key = bytes(range(32))
    data = b"This is a test message for Threefish cipher!"

    ct = opt_enc(data, key)
    pt = opt_dec(ct, key)

    print("\n[OPTIMIZED]")
    print("Ciphertext:", ct.hex())
    print("Plain text:", pt)

    assert pt == data
    print("Optimized test passed!")

def test_threefish_numba():
    key = bytes(range(32))
    data = b"This is a test message for Threefish cipher!"

    ct = numba_enc(data, key)
    pt = numba_dec(ct, key)

    print("\n[NUMBA]")
    print("Ciphertext:", ct.hex())
    print("Plain text:", pt)

    assert pt == data
    print("Numba test passed!")

def test_threefish_gpu():
    key = bytes(range(32))
    data = b"This is a test message for Threefish cipher!"

    ct = gpu_enc(data, key)
    pt = gpu_dec(ct, key)

    print("\n[OPTIMIZED]")
    print("Ciphertext:", ct.hex())
    print("Plain text:", pt)

    assert pt == data
    print("GPU test passed!")

def test_consistency():
    key = bytes(range(32))
    data = b"Consistency check across implementations!"

    ct1 = naive_enc(data, key)
    ct2 = opt_enc(data, key)
    ct3 = numba_enc(data, key)
    ct4 = gpu_enc(data,key)

    assert ct1 == ct2 == ct3 == ct4
    print("\n[CONSISTENCY CHECK PASSED]")


if __name__ == "__main__":
    test_threefish_naive()
    test_threefish_optimized()
    test_threefish_numba()
    test_threefish_gpu()
    test_consistency()