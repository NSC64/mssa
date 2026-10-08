#!/usr/bin/env python3
"""Prepare and run the four opt-in MSSA research studies.

The script deliberately keeps raw JSON outside the repository by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SEEDS = [7401, 7402, 7403, 7404, 7405]
METHODS = ["adamw", "interdiffusion", "interdiffusion_recurrent_only", "readout_only"]
RATES = [0.001, 0.003, 0.01]


def split_corpus(source: Path, destination: Path) -> dict[str, object]:
    raw = source.read_bytes()
    if len(raw) < 1024:
        raise SystemExit("corpus is too small for the study")
    destination.mkdir(parents=True, exist_ok=True)
    cuts = [0, len(raw) * 8 // 10, len(raw) * 9 // 10, len(raw)]
    names = ["train.bin", "dev.bin", "test.bin"]
    for name, start, end in zip(names, cuts, cuts[1:]):
        path = destination / name
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_bytes(raw[start:end])
    return {
        "source": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "split_bytes": dict(zip(names, (end - start for start, end in zip(cuts, cuts[1:])))),
    }


def worker(binary: Path, args: list[str], output: Path) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(binary), *args, str(output)], check=True)


def language(binary: Path, split: Path, output: Path, updates: int) -> None:
    for method in METHODS:
        rates = RATES if method in {"adamw", "interdiffusion"} else [0.003]
        for rate in rates:
            for seed in SEEDS:
                worker(
                    binary,
                    [
                        "language",
                        method,
                        str(split / "train.bin"),
                        str(split / "dev.bin"),
                        str(split / "test.bin"),
                        str(updates),
                        str(seed),
                        str(rate),
                    ],
                    output / f"language-{method}-{rate:g}-{seed}.json",
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/research_studies"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("__agent__/research_studies"))
    parser.add_argument("--updates", type=int, default=1024)
    args = parser.parse_args()
    if args.updates <= 0:
        raise SystemExit("--updates must be positive")
    if not args.binary.is_file():
        raise SystemExit(f"missing study binary: {args.binary}")
    split = args.out / "corpus"
    manifest = split_corpus(args.corpus, split)
    (args.out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    language(args.binary, split, args.out / "language", args.updates)
    for seed in SEEDS:
        worker(args.binary, ["timing", str(seed), str(args.updates * 2)], args.out / "timing" / f"{seed}.json")
        worker(args.binary, ["sparse", str(seed)], args.out / "sparse" / f"{seed}.json")
        worker(args.binary, ["dream", str(seed), str(max(32, args.updates // 16))], args.out / "dream" / f"{seed}.json")


if __name__ == "__main__":
    main()
