from LEA.lea_naive import encrypt as naive_enc, decrypt as naive_dec
from LEA.lea_optimized import encrypt as opt_enc, decrypt as opt_dec
from LEA.lea_numba import encrypt as numba_enc, decrypt as numba_dec
from LEA.lea_gpu import encrypt as gpu_enc

# -----------------------------
# Basic Test
# -----------------------------

def test_lea_naive():
    key = bytes(range(16))
    data = b"This is a test message for LEA cipher!"

    ct = naive_enc(data, key)
    pt = naive_dec(ct, key)

    print("\n[LEA-Naive]")
    print("Ciphertext:", ct.hex())
    print("Recovered :", pt)

    assert pt == data
    print("LEA Naive test passed!")

def test_lea_opt():
    key = bytes(range(16))
    data = b"This is a test message for LEA cipher!"

    ct = opt_enc(data, key)
    pt = opt_dec(ct, key)

    print("\n[LEA-Optimized]")
    print("Ciphertext:", ct.hex())
    print("Recovered :", pt)

    assert pt == data
    print("LEA Optimized test passed!")

def test_lea_numba():
    key = bytes(range(16))
    data = b"This is a test message for LEA cipher!"

    ct = numba_enc(data, key)
    pt = numba_dec(ct, key)

    print("\n[LEA-NUMBA]")
    print("Ciphertext:", ct.hex())
    print("Recovered :", pt)

    assert pt == data
    print("LEA Numba test passed!")

def test_lea_cuda():
    key = bytes(range(16))
    data = b"This is a test message for LEA cipher!"

    ct = gpu_enc(data, key)
    pt = numba_dec(ct, key) # reusing decryption implementation of numba 

    print("\n[LEA-CUDA]")
    print("Ciphertext:", ct.hex())
    print("Recovered :", pt)

    assert pt == data
    print("LEA Cuda test passed!")

# -----------------------------
# Consistency Test (future ready)
# -----------------------------

def test_lea_consistency():
    key = bytes(range(16))
    data = b"Consistency check across LEA implementations!"

    ct_naive = naive_enc(data, key)
    ct_opt = opt_enc(data, key)
    ct_numba = numba_enc(data, key)
    ct_gpu = gpu_enc(data, key)

    print("\n[CONSISTENCY CHECK]")
    print("Naive CT     :", ct_naive.hex())
    print("Optimized CT :", ct_opt.hex())
    print("Numba CT     :", ct_numba.hex())
    print("CUDA Ct      :", ct_gpu.hex())

    # Only naive for now (add others later)
    assert naive_dec(ct_naive, key) == data
    assert opt_dec(ct_opt, key) == data
    assert numba_dec(ct_numba, key) == data
    assert numba_dec(ct_gpu, key) == data

    print("Consistency test passed!")


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    test_lea_naive()
    test_lea_opt()
    test_lea_numba()
    test_lea_cuda()
    test_lea_consistency()