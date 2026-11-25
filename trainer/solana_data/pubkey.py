"""Minimal pure-Python Solana public key helpers (base58, PDA derivation).

Implemented here so the trainer has no native-extension dependencies.
"""
from __future__ import annotations

import hashlib

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        n = n * 58 + _B58_INDEX[ch]
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + raw


def b58encode(b: bytes) -> str:
    n = int.from_bytes(b, "big")
    out = []
    while n:
        n, r = divmod(n, 58)
        out.append(_B58_ALPHABET[r])
    pad = len(b) - len(b.lstrip(b"\x00"))
    return "1" * pad + "".join(reversed(out))


# --- ed25519 on-curve check (needed to derive program addresses) -----------
_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P


def is_on_curve(key: bytes) -> bool:
    """True if the 32 bytes decompress to a valid ed25519 point."""
    y = int.from_bytes(key, "little") & ((1 << 255) - 1)
    sign = key[31] >> 7
    if y >= _P:
        return False
    y2 = y * y % _P
    u = (y2 - 1) % _P
    v = (_D * y2 + 1) % _P
    x2 = u * pow(v, _P - 2, _P) % _P
    if x2 == 0:
        return sign == 0
    return pow(x2, (_P - 1) // 2, _P) == 1


def create_program_address(seeds: list[bytes], program_id: str) -> str | None:
    h = hashlib.sha256()
    for s in seeds:
        h.update(s)
    h.update(b58decode(program_id))
    h.update(b"ProgramDerivedAddress")
    digest = h.digest()
    if is_on_curve(digest):
        return None
    return b58encode(digest)


def find_program_address(seeds: list[bytes], program_id: str) -> tuple[str, int]:
    for bump in range(255, -1, -1):
        addr = create_program_address(seeds + [bytes([bump])], program_id)
        if addr is not None:
            return addr, bump
    raise ValueError("Unable to find a viable program address bump seed")


def associated_token_address(owner: str, mint: str, token_program: str) -> str:
    from .constants import ATA_PROGRAM

    addr, _ = find_program_address(
        [b58decode(owner), b58decode(token_program), b58decode(mint)], ATA_PROGRAM
    )
    return addr


def is_pda(address: str) -> bool:
    """Program-derived addresses are off the ed25519 curve (no private key)."""
    return not is_on_curve(b58decode(address).rjust(32, b"\x00"))
