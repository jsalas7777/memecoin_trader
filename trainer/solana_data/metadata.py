"""Read token name / symbol / uri from chain (Token-2022 extension or Metaplex)."""
from __future__ import annotations

import base64
import struct

from .constants import METAPLEX_METADATA_PROGRAM
from .pubkey import b58decode, find_program_address
from .rpc import SolanaRPC


def _read_str(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    return buf[off : off + n].decode("utf-8", "replace").rstrip("\x00").strip(), off + n


async def get_metadata(rpc: SolanaRPC, mint: str, token: dict | None = None) -> dict:
    for ext in (token or {}).get("extensions", []):
        if ext.get("extension") == "tokenMetadata":
            st = ext["state"]
            return {"name": st.get("name", ""), "symbol": st.get("symbol", ""), "uri": st.get("uri", "")}
    pda, _ = find_program_address(
        [b"metadata", b58decode(METAPLEX_METADATA_PROGRAM), b58decode(mint)], METAPLEX_METADATA_PROGRAM
    )
    acc = (await rpc.get_multiple_accounts([pda], encoding="base64"))[0]
    if acc is None:
        return {"name": "", "symbol": "", "uri": ""}
    buf = base64.b64decode(acc["data"][0])
    off = 1 + 32 + 32  # key, update_authority, mint
    try:
        name, off = _read_str(buf, off)
        symbol, off = _read_str(buf, off)
        uri, off = _read_str(buf, off)
    except struct.error:
        return {"name": "", "symbol": "", "uri": ""}
    return {"name": name, "symbol": symbol, "uri": uri}
