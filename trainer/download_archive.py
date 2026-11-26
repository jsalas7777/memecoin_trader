#!/usr/bin/env python3
"""Download a public pump.fun trade archive from Hugging Face into trainer/archives/.

  python download_archive.py              # 1-day sample: every pump.fun trade of 2026-10-15 (~1.5 GB)
  python download_archive.py corpus       # 39-day corpus, Jun-Jul 2026 (~6.7 GB)
  python download_archive.py --list       # what's available, sizes, what's already downloaded

  archives/<name>/   the files, exactly as published (archives/ is in .gitignore, never committed)

Resumable: if it stops (Ctrl+C, network), run the same command again and it continues;
files already downloaded are skipped. Checks free disk space first, and at the end opens
every Parquet table and prints its row count, so a broken download is noticed immediately.
No account or key needed (public datasets).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVES = HERE / "archives"

DATASETS = {
    "sample": {
        "repo": "pumpdatafun/pump-fun-trades-sample",
        "folder": "pumpdata-2026-10-15",
        "about": "1 UTC day (2026-10-15): every pump.fun curve + PumpSwap trade (21.8M landed), launches, "
                 "graduations, events. From the chain's ledger. License: MIT",
        "exclude": ["signatures/*"],  # 3.1 GB of transaction signatures, not needed for training
    },
    "corpus": {
        "repo": "Slinky21/Pumpfun_Memecoin_Corpus",
        "folder": "pumpfun-corpus-2026-06",
        "about": "39 days (Jun-Jul 2026): 798,430 launches, 33.6M trades, 26.9M curve snapshots, graduation "
                 "labels. Read KNOWN_ISSUES.md first. License: CC BY 4.0 (credit: Slink21)",
        "exclude": [],
    },
}


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def remote_files(repo: str, exclude: list[str]) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in info.siblings
            if not any(fnmatch(s.rfilename, p) for p in exclude)]


def local_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and ".cache" not in p.parts) \
        if folder.exists() else 0


def verify(folder: Path) -> bool:
    import pyarrow.parquet as pq
    ok = True
    print("\nChecking the downloaded tables:")
    tables: dict[str, list[Path]] = {}
    for p in sorted(folder.rglob("*.parquet")):
        if ".cache" in p.parts:
            continue
        rel = p.relative_to(folder)
        tables.setdefault(str(rel.parent) if str(rel.parent) != "." else p.stem, []).append(p)
    for name, files in tables.items():
        try:
            rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            cols = pq.ParquetFile(files[0]).schema_arrow.names
            print(f"  ✓ {name:<28} {rows:>14,} rows  ({len(files)} file{'s' if len(files) > 1 else ''}, "
                  f"{len(cols)} columns)")
        except Exception as e:  # noqa: BLE001 - report every broken file, keep checking the rest
            ok = False
            print(f"  ✗ {name}: unreadable ({e}). Run the same command again to re-download it.")
    return ok


def cmd_list() -> None:
    for name, d in DATASETS.items():
        folder = ARCHIVES / d["folder"]
        try:
            total = sum(s for _, s in remote_files(d["repo"], d["exclude"]))
            size = gb(total)
        except Exception as e:  # noqa: BLE001
            total, size = 0, f"? ({type(e).__name__})"
        have = local_size(folder)
        state = "not downloaded" if not have else ("downloaded" if total and have >= total * 0.99 else
                                                    f"partial ({gb(have)})")
        print(f"{name:<8} {size:>9}  {state:<22} {d['repo']}\n         {d['about']}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="sample", choices=list(DATASETS),
                   help="sample (default, ~1.5 GB) or corpus (~6.7 GB)")
    p.add_argument("--list", action="store_true", help="show the available archives and what's downloaded")
    args = p.parse_args()
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        raise SystemExit("missing package: run  .venv/bin/pip install -r requirements.txt")
    if args.list:
        return cmd_list()

    d = DATASETS[args.dataset]
    folder = ARCHIVES / d["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Dataset : {d['repo']}\n          {d['about']}\nSaving to: {folder}")

    files = remote_files(d["repo"], d["exclude"])
    total = sum(s for _, s in files)
    have = local_size(folder)
    need = max(0, total - have)
    free = shutil.disk_usage(folder).free
    print(f"Size    : {gb(total)} in {len(files)} files" + (f" ({gb(have)} already here)" if have else ""))
    print(f"Disk    : {gb(free)} free")
    if need > free - 2e9:  # keep 2 GB spare
        raise SystemExit(f"✗ Not enough disk space: need {gb(need)} more, only {gb(free)} free. "
                         "Free some space and run again.")
    if d["exclude"]:
        print(f"Skipping: {', '.join(d['exclude'])} (not needed for training)")
    print()

    t0 = time.time()
    for attempt in range(1, 6):
        try:
            snapshot_download(repo_id=d["repo"], repo_type="dataset", local_dir=folder,
                              ignore_patterns=d["exclude"] or None, max_workers=4)
            break
        except KeyboardInterrupt:
            raise SystemExit("\nStopped. Run the same command again to continue where it left off.")
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            wait = 60 if code == 429 else 10 * attempt
            what = "Hugging Face rate limit (HTTP 429)" if code == 429 else f"download error ({code or e})"
            if attempt == 5:
                raise SystemExit(f"✗ {what}; gave up after 5 tries. Run the same command again later.")
            print(f"⚠ {time.strftime('%H:%M:%S')} {what}: retrying in {wait}s (try {attempt + 1}/5)")
            time.sleep(wait)
        except OSError as e:
            if attempt == 5:
                raise SystemExit(f"✗ network/disk error: {e}. Run the same command again to resume.")
            print(f"⚠ {time.strftime('%H:%M:%S')} network error ({e}): retrying in {10 * attempt}s")
            time.sleep(10 * attempt)

    print(f"\n✓ Downloaded {gb(local_size(folder))} in {time.time() - t0:.0f}s -> {folder}")
    if not verify(folder):
        sys.exit(1)
    if args.dataset == "corpus":
        print(f"\nRead {folder / 'KNOWN_ISSUES.md'} before using this data.")


if __name__ == "__main__":
    main()
