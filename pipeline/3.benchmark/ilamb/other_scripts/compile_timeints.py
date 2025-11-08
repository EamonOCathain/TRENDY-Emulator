#!/usr/bin/env python3
"""
Collect ILAMB global time-integral and bias images per variable & scenario.

Input layout (discovered automatically):
  <in_dir>/<S*/>/_build/<Category>/<VarLong>/ENSMEAN/*.png

We copy up to 5 images (when present) into:
  <out_dir>/<VarShort>/<S*/>/

Saved filenames:
  Benchmark_global_timeint.png
  <model>_global_timeint.png   (model file; original basename preserved)
  legend_timeint.png
  <model>_global_bias.png      (model file; original basename preserved)
  legend_bias.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import re

# Map ILAMB variable long names (directory names) to short codes
VAR_MAP = {
    # Carbon cycle (fluxes)
    "GrossPrimaryProduction": "gpp",
    "NetPrimaryProduction": "npp",
    "AutotrophicRespiration": "ra",
    "HeterotrophicRespiration": "rh",
    "NetBiomeProductivity": "nbp",
    "FireEmissions": "fire",
    "Land-UseChangeEmissions": "luc",

    # Carbon stocks
    "CarboninVegetation": "cVeg",
    "CarboninSoilPool": "cSoil",
    "CarboninLitterPool": "cLitter",
    "CarboninEcosystem": "cTotal",

    # Vegetation structure
    "LeafAreaIndex": "lai",

    # Water cycle
    "Evapotranspiration": "evap",
    "TotalSoilMoistureContent": "mrso",
    "TotalRunoff": "mrro",
}

def slugify(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s or "var"

def first_match(paths: list[Path]) -> Path | None:
    return sorted(paths)[0] if paths else None

def find_benchmark_timeint(en_dir: Path) -> Path | None:
    # Prefer exact canonical name; fall back to any Benchmark_*global_timeint.png
    p = en_dir / "Benchmark_global_timeint.png"
    if p.exists():
        return p
    return first_match([q for q in en_dir.glob("Benchmark_*global_timeint.png") if q.is_file()])

def find_model_timeint(en_dir: Path) -> Path | None:
    # Any *global_timeint.png excluding Benchmark_* and legend_*
    cands = []
    for q in en_dir.glob("*global_timeint.png"):
        name = q.name
        if name.startswith("Benchmark_"):
            continue
        if name.startswith("legend_"):
            continue
        if q.is_file():
            cands.append(q)
    return first_match(cands)

def find_model_bias(en_dir: Path) -> Path | None:
    # Any *global_bias.png excluding Benchmark_* and legend_*
    cands = []
    for q in en_dir.glob("*global_bias.png"):
        name = q.name
        if name.startswith("Benchmark_"):
            continue
        if name.startswith("legend_"):
            continue
        if q.is_file():
            cands.append(q)
    return first_match(cands)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", type=Path, help="Root directory containing scenario folders (e.g., S0, S1, ...)")
    ap.add_argument("out_dir", type=Path, help="Destination root for collected images")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in out_dir")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_root: Path = args.out_dir
    overwrite: bool = args.overwrite

    if not in_dir.exists():
        print(f"[ERR] in_dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(2)

    # Discover scenarios: immediate children like S0, S1, S2, S3...
    scenarios = [p for p in sorted(in_dir.iterdir()) if p.is_dir() and p.name.startswith("S")]
    if not scenarios:
        print(f"[WARN] No scenario directories (e.g., S0) found under {in_dir}", file=sys.stderr)

    total_copied = 0
    summary = []

    for scen_dir in scenarios:
        scenario = scen_dir.name
        build_dir = scen_dir / "_build"
        if not build_dir.exists():
            print(f"[WARN] No _build directory for {scenario}: {build_dir}")
            continue

        # Find all ENSEMBLE result folders: .../_build/<Category>/<VarLong>/ENSMEAN/
        en_paths = sorted(build_dir.glob("*/*/ENSMEAN"))
        if not en_paths:
            print(f"[WARN] No ENSMEAN directories found under {build_dir}")
            continue

        scen_copied = 0
        vars_seen = set()

        for en_dir in en_paths:
            var_long = en_dir.parent.name         # e.g., "GrossPrimaryProduction"
            _category = en_dir.parent.parent.name  # e.g., "CarbonCycle"
            var_short = VAR_MAP.get(var_long, slugify(var_long))

            bench_timeint = find_benchmark_timeint(en_dir)
            model_timeint = find_model_timeint(en_dir)
            legend_timeint = en_dir / "legend_timeint.png"
            model_bias = find_model_bias(en_dir)
            legend_bias = en_dir / "legend_bias.png"  # per user note name is legend_bias.png

            # Prepare output directory
            out_dir = out_root / var_short / scenario
            out_dir.mkdir(parents=True, exist_ok=True)

            def _copy_if_exists(src: Path | None, dst_name: str) -> int:
                if src and src.exists():
                    dst = out_dir / dst_name
                    if dst.exists() and not overwrite:
                        return 0
                    shutil.copy2(src, dst)
                    return 1
                return 0

            copied = 0
            # Preserve original model filenames for clarity; use canonical names for benchmarks/legends
            copied += _copy_if_exists(bench_timeint, "Benchmark_global_timeint.png")
            copied += _copy_if_exists(model_timeint, model_timeint.name if model_timeint else "Model_global_timeint.png")
            copied += _copy_if_exists(legend_timeint, "legend_timeint.png")
            copied += _copy_if_exists(model_bias, model_bias.name if model_bias else "Model_global_bias.png")
            copied += _copy_if_exists(legend_bias, "legend_bias.png")

            # Helpful diagnostics if something is missing
            if copied == 0:
                print(f"[INFO] No copies for {scenario}/{var_short} (check files in {en_dir})")
            else:
                if not bench_timeint or not bench_timeint.exists():
                    print(f"[MISS] {scenario}/{var_short}: Benchmark_global_timeint.png not found")
                if not model_timeint or not model_timeint.exists():
                    print(f"[MISS] {scenario}/{var_short}: model *global_timeint.png not found")
                if not model_bias or not model_bias.exists():
                    print(f"[MISS] {scenario}/{var_short}: model *global_bias.png not found")
                if not legend_timeint.exists():
                    print(f"[MISS] {scenario}/{var_short}: legend_timeint.png not found")
                if not legend_bias.exists():
                    print(f"[MISS] {scenario}/{var_short}: legend_bias.png not found")

                scen_copied += copied
                total_copied += copied
                vars_seen.add(var_short)

        summary.append((scenario, len(vars_seen), scen_copied))

    # Print summary
    print("\n=== Copy Summary ===")
    for scenario, nvars, nfiles in summary:
        print(f"{scenario}: {nvars} variables, {nfiles} files copied")
    print(f"TOTAL: {total_copied} files copied into {out_root}")
    if not overwrite:
        print("(Use --overwrite to replace existing files.)")

if __name__ == "__main__":
    main()