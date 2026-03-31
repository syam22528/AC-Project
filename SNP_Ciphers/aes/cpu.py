"""CPU AES-128 implementation (software/Numba only).

Single-file design:
- Software path: clean no-T-table AES rounds (Numba + Python fallback)
"""

from __future__ import annotations

import os

import numpy as np

from ctr_utils import build_ctr_blocks, xor_bytes

try:
	from numba import get_num_threads, njit, prange, set_num_threads

	NUMBA_AVAILABLE = True
except Exception:
	NUMBA_AVAILABLE = False

	def njit(*args, **kwargs):  # type: ignore
		def _decorator(fn):
			return fn

		return _decorator

	def prange(*args):  # type: ignore
		return range(*args)

	def set_num_threads(n: int) -> None:  # type: ignore
		return None

	def get_num_threads() -> int:  # type: ignore
		return 1




SBOX = np.array(
	[
		0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
		0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
		0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
		0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
		0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
		0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
		0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
		0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
		0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
		0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
		0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
		0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
		0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
		0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
		0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
		0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
	],
	dtype=np.uint8,
)
RCON = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], dtype=np.uint8)


@njit(cache=True)
def _xtime(x: int) -> int:
	return ((x << 1) ^ (0x1B if (x & 0x80) else 0x00)) & 0xFF


@njit(cache=True)
def _encrypt_component_jit(data: np.ndarray, expanded: np.ndarray, sbox: np.ndarray) -> np.ndarray:
	out = np.empty(data.size, dtype=np.uint8)

	for base in range(0, data.size, 16):
		s = np.empty(16, dtype=np.uint8)
		for i in range(16):
			s[i] = data[base + i] ^ expanded[i]

		for rnd in range(1, 10):
			for i in range(16):
				s[i] = sbox[np.int64(s[i])]

			t1, t5, t9, t13 = s[1], s[5], s[9], s[13]
			s[1], s[5], s[9], s[13] = t5, t9, t13, t1
			t2, t6, t10, t14 = s[2], s[6], s[10], s[14]
			s[2], s[6], s[10], s[14] = t10, t14, t2, t6
			t3, t7, t11, t15 = s[3], s[7], s[11], s[15]
			s[3], s[7], s[11], s[15] = t15, t3, t7, t11

			for c in range(4):
				i = c * 4
				a0, a1, a2, a3 = s[i], s[i + 1], s[i + 2], s[i + 3]
				t = a0 ^ a1 ^ a2 ^ a3
				u = a0
				s[i] ^= t ^ _xtime(a0 ^ a1)
				s[i + 1] ^= t ^ _xtime(a1 ^ a2)
				s[i + 2] ^= t ^ _xtime(a2 ^ a3)
				s[i + 3] ^= t ^ _xtime(a3 ^ u)

			rbase = rnd * 16
			for i in range(16):
				s[i] ^= expanded[rbase + i]

		for i in range(16):
			s[i] = sbox[np.int64(s[i])]

		t1, t5, t9, t13 = s[1], s[5], s[9], s[13]
		s[1], s[5], s[9], s[13] = t5, t9, t13, t1
		t2, t6, t10, t14 = s[2], s[6], s[10], s[14]
		s[2], s[6], s[10], s[14] = t10, t14, t2, t6
		t3, t7, t11, t15 = s[3], s[7], s[11], s[15]
		s[3], s[7], s[11], s[15] = t15, t3, t7, t11

		for i in range(16):
			out[base + i] = s[i] ^ expanded[160 + i]

	return out


@njit(cache=True, parallel=True)
def _encrypt_component_jit_parallel(data: np.ndarray, expanded: np.ndarray, sbox: np.ndarray) -> np.ndarray:
	out = np.empty(data.size, dtype=np.uint8)
	nblocks = data.size // 16

	for b in prange(nblocks):
		base = b * 16
		s = np.empty(16, dtype=np.uint8)
		for i in range(16):
			s[i] = data[base + i] ^ expanded[i]

		for rnd in range(1, 10):
			for i in range(16):
				s[i] = sbox[np.int64(s[i])]

			t1, t5, t9, t13 = s[1], s[5], s[9], s[13]
			s[1], s[5], s[9], s[13] = t5, t9, t13, t1
			t2, t6, t10, t14 = s[2], s[6], s[10], s[14]
			s[2], s[6], s[10], s[14] = t10, t14, t2, t6
			t3, t7, t11, t15 = s[3], s[7], s[11], s[15]
			s[3], s[7], s[11], s[15] = t15, t3, t7, t11

			for c in range(4):
				i = c * 4
				a0, a1, a2, a3 = s[i], s[i + 1], s[i + 2], s[i + 3]
				t = a0 ^ a1 ^ a2 ^ a3
				u = a0
				s[i] ^= t ^ _xtime(a0 ^ a1)
				s[i + 1] ^= t ^ _xtime(a1 ^ a2)
				s[i + 2] ^= t ^ _xtime(a2 ^ a3)
				s[i + 3] ^= t ^ _xtime(a3 ^ u)

			rbase = rnd * 16
			for i in range(16):
				s[i] ^= expanded[rbase + i]

		for i in range(16):
			s[i] = sbox[np.int64(s[i])]

		t1, t5, t9, t13 = s[1], s[5], s[9], s[13]
		s[1], s[5], s[9], s[13] = t5, t9, t13, t1
		t2, t6, t10, t14 = s[2], s[6], s[10], s[14]
		s[2], s[6], s[10], s[14] = t10, t14, t2, t6
		t3, t7, t11, t15 = s[3], s[7], s[11], s[15]
		s[3], s[7], s[11], s[15] = t15, t3, t7, t11

		for i in range(16):
			out[base + i] = s[i] ^ expanded[160 + i]

	return out



class AesCpuOptimized:
	"""AES-128 CPU facade (software/Numba only).

	Args:
		use_numba: enable Numba acceleration for software path
	"""

	def __init__(self, use_numba: bool = True) -> None:
		self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
		self._sw_last_key: bytes | None = None
		self._expanded_np: np.ndarray | None = None

	@staticmethod
	def _validate_inputs(data: bytes, key: bytes) -> None:
		if len(key) != 16:
			raise ValueError("AES-128 requires a 16-byte key")
		if len(data) % 16 != 0:
			raise ValueError("ECB input length must be a multiple of 16 bytes")

	@staticmethod
	def _key_expansion_128(key: bytes) -> list[int]:
		expanded = list(key) + [0] * (176 - 16)
		bytes_generated = 16
		rcon_iter = 0
		while bytes_generated < 176:
			t0 = expanded[bytes_generated - 4]
			t1 = expanded[bytes_generated - 3]
			t2 = expanded[bytes_generated - 2]
			t3 = expanded[bytes_generated - 1]
			if bytes_generated % 16 == 0:
				t0, t1, t2, t3 = t1, t2, t3, t0
				t0, t1, t2, t3 = int(SBOX[t0]), int(SBOX[t1]), int(SBOX[t2]), int(SBOX[t3])
				t0 ^= int(RCON[rcon_iter])
				rcon_iter += 1
			expanded[bytes_generated + 0] = expanded[bytes_generated - 16] ^ t0
			expanded[bytes_generated + 1] = expanded[bytes_generated - 15] ^ t1
			expanded[bytes_generated + 2] = expanded[bytes_generated - 14] ^ t2
			expanded[bytes_generated + 3] = expanded[bytes_generated - 13] ^ t3
			bytes_generated += 4
		return expanded

	def _prepare_key(self, key: bytes) -> np.ndarray:
		if self._sw_last_key != key or self._expanded_np is None:
			self._expanded_np = np.array(self._key_expansion_128(key), dtype=np.uint8)
			self._sw_last_key = bytes(key)
		return self._expanded_np



	def _encrypt_software_ecb(self, data: bytes, key: bytes, workers: int) -> bytes:
		if not self.use_numba:
			raise RuntimeError("Numba JIT is required for AES software CPU encryption")

		expanded_np = self._prepare_key(key)
		data_np = np.frombuffer(data, dtype=np.uint8)

		if workers > 1:
			cpu_count = os.cpu_count() or 1
			w = max(1, min(workers, cpu_count))
			old_threads = get_num_threads()
			try:
				set_num_threads(w)
				out_np = _encrypt_component_jit_parallel(data_np, expanded_np, SBOX)
			finally:
				set_num_threads(old_threads)
		else:
			out_np = _encrypt_component_jit(data_np, expanded_np, SBOX)
		return out_np.tobytes()

	def encrypt_ecb(self, data: bytes, key: bytes, workers: int = 1) -> bytes:
		self._validate_inputs(data, key)
		if len(data) == 0:
			return b""
		return self._encrypt_software_ecb(data, key, workers=workers)

	def encrypt_ctr(self, data: bytes, key: bytes, workers: int = 1, nonce: bytes | None = None) -> bytes:
		"""Encrypt data in CTR mode using ECB(counter) keystream generation."""
		if len(key) != 16:
			raise ValueError("AES-128 requires a 16-byte key")
		if len(data) % 16 != 0:
			raise ValueError("CTR input length must be a multiple of 16 bytes")
		if len(data) == 0:
			return b""

		nblocks = len(data) // 16
		ctr_blocks = build_ctr_blocks(nblocks, 16, nonce=nonce)
		keystream = self.encrypt_ecb(ctr_blocks, key, workers=workers)
		return xor_bytes(data, keystream)


__all__ = ["AesCpuOptimized"]
