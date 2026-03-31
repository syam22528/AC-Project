from CHAM.cham_naive import encrypt as naive_enc, decrypt as naive_dec
from CHAM.cham_optimized import encrypt as opt_enc, decrypt as opt_dec
from CHAM.cham_numba import encrypt as numba_enc, decrypt as numba_dec
from CHAM.cham_gpu import encrypt as gpu_enc, decrypt as gpu_dec

def test_cham_naive():
    key = bytes(range(16))
    data = b"This is a test message for CHAM cipher!"

    ct = naive_enc(data, key)
    pt = naive_dec(ct, key)

    print("\n[NAIVE]")
    print("Ciphertext:", ct.hex())
    print("Plain text: ", pt)

    assert pt == data
    print("Naive test passed!")

def test_cham_optmized():
    key = bytes(range(16))
    data = b"This is a test message for CHAM cipher!"

    ct = opt_enc(data, key)
    pt = opt_dec(ct, key)

    print("\n[OPTIMIZED]")
    print("Ciphertext:", ct.hex())
    print("Plain text: ", pt)

    assert pt == data
    print("Optimized test passed!")

def test_cham_numba():
    key = bytes(range(16))
    data = b"This is a test message for CHAM cipher!"

    ct = numba_enc(data, key)
    pt = numba_dec(ct, key)

    print("\n[NUMBA]")
    print("Ciphertext:", ct.hex())
    print("Plain text: ", pt)

    assert pt == data
    print("Numba test passed!")

def test_cham_gpu():
    key = bytes(range(16))
    data = b"This is a test message for CHAM cipher!"

    ct = gpu_enc(data, key)
    pt = gpu_dec(ct, key)

    print("\n[GPU]")
    print("Ciphertext:", ct.hex())
    print("Plain text: ", pt)

    assert pt == data
    print("GPU test passed!")

def test_cham_consistency():
    key = bytes(range(16))
    data = b"Consistency check across CHAM implementations!"

    ct_naive = naive_enc(data, key)
    ct_opt = opt_enc(data, key)
    ct_numba = numba_enc(data, key)
    ct_gpu = gpu_enc(data, key)

    print("\n[CONSISTENCY CHECK]")
    print("Naive CT     :", ct_naive.hex())
    print("Optimized CT :", ct_opt.hex())
    print("Numba CT     :", ct_numba.hex())
    print("CUDA Ct      :", ct_gpu.hex())

    assert naive_dec(ct_naive, key) == data
    assert opt_dec(ct_opt, key) == data
    assert numba_dec(ct_numba, key) == data
    assert gpu_dec(ct_gpu, key) == data
    assert ct_naive == ct_opt == ct_numba == ct_gpu

    print("Consistency test passed!")
if __name__ == '__main__':
    test_cham_naive()
    test_cham_optmized()
    test_cham_numba()
    test_cham_gpu()
    test_cham_consistency()