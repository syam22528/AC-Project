from SPECK.speck_naive import encrypt as naive_enc
from SPECK.speck_naive import decrypt as naive_dec

from SPECK.speck_optimized import encrypt as opt_enc
from SPECK.speck_optimized import decrypt as opt_dec

from SPECK.speck_numba import encrypt as numba_enc
from SPECK.speck_numba import decrypt as numba_dec

from SPECK.speck_gpu import encrypt as gpu_enc
from SPECK.speck_gpu import decrypt as gpu_dec

def test_speck_naive():
    key = bytes(range(16))
    data = b"This is a test message for SPECK cipher!"

    ct = naive_enc(data, key)
    pt = naive_dec(ct, key)

    print("\n[NAIVE]")
    print("Ciphertext:", ct.hex())

    assert pt[:len(data)] == data
    print("Naive test passed!")

def test_speck_optimized():
    key = bytes(range(16))
    data = b"This is a test message for optimized SPECK!"

    ct = opt_enc(data, key)
    pt = opt_dec(ct, key)

    print("\n[OPTIMIZED]")
    print("Ciphertext:", ct.hex())
    print(pt)
    assert pt[:len(data)] == data
    print("Optimized test passed!")

def test_speck_numba():
    key = bytes(range(16))
    data = b"This is a test message for numba SPECK!"

    ct = numba_enc(data, key)
    pt = numba_dec(ct, key)

    print("\n[NUMBA]")
    print("Ciphertext:", ct.hex())

    assert pt[:len(data)] == data
    print("NUMBA test passed!")

def test_speck_cuda():
    key = bytes(range(16))
    data = b"This is a test message for numba SPECK!"

    ct = gpu_enc(data, key)
    pt = gpu_dec(ct, key)

    print("\n[CUDA]")
    print("Ciphertext:", ct.hex())
    print(pt)

    assert pt[:len(data)] == data
    print("CUDA test passed!")


def test_speck_consistency():
    key = bytes(range(16))
    data = b"Consistency check SPECK!"

    ct_naive = naive_enc(data, key)
    ct_opt = opt_enc(data, key)
    ct_numba = numba_enc(data, key)
    ct_gpu = gpu_enc(data, key)

    print("\n[CONSISTENCY]")
    print("Naive    :", ct_naive.hex())
    print("Opt      :", ct_opt.hex())
    print("Numba    :", ct_numba.hex())
    print("CUDA     :", ct_gpu.hex())

    assert naive_dec(ct_naive, key)[:len(data)] == data
    assert opt_dec(ct_opt, key)[:len(data)] == data
    assert numba_dec(ct_numba, key)[:len(data)] == data
    assert gpu_dec(ct_gpu, key)[:len(data)] == data

    print("Consistency test passed!")


if __name__ == "__main__":
    test_speck_naive()
    test_speck_optimized()
    test_speck_numba()
    test_speck_cuda()
    test_speck_consistency()