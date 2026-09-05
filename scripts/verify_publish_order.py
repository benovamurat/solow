#!/usr/bin/env python3
"""Verify that `scripts/publish.sh`'s ORDER list is a valid leaf-to-root
topological sort of the workspace's Cargo.toml dependency graph.

Usage:
    python3 scripts/verify_publish_order.py         # print current + expected
    python3 scripts/verify_publish_order.py --check # exit non-zero if drift

The workspace has 30+ inter-dependent crates; a wrong publish order fails
mid-way with the crates.io index rejecting a dependency because it was
uploaded later than its dependent. Running this script before every
release (or wiring it into CI) catches that class of bug before you paste
your token.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATES = ROOT / "crates"
PUBLISH_SH = ROOT / "scripts" / "publish.sh"
NON_PUBLISHED = {"solow-py", "solow-polars", "solow-bench", "solow-gallery"}


def read_deps() -> dict[str, set[str]]:
    """Return {crate: {solow-* deps}} for every publishable crate."""
    deps: dict[str, set[str]] = {}
    for d in sorted(CRATES.iterdir()):
        tf = d / "Cargo.toml"
        if not tf.exists():
            continue
        txt = tf.read_text()
        if re.search(r"^publish\s*=\s*false", txt, re.M):
            continue
        name_m = re.search(r'^name\s*=\s*"([^"]+)"', txt, re.M)
        if not name_m:
            continue
        name = name_m.group(1)
        ds: set[str] = set()
        for line in txt.splitlines():
            m = re.match(r"^(solow-[a-z0-9-]+)\s*[.=]", line)
            if m:
                dep = m.group(1)
                if dep != name and dep not in NON_PUBLISHED:
                    ds.add(dep)
        deps[name] = ds
    return deps


def topo_sort(deps: dict[str, set[str]]) -> list[str]:
    """Kahn's algorithm with deterministic alphabetic tie-break."""
    remaining = {n: set(d) for n, d in deps.items()}
    order: list[str] = []
    while remaining:
        ready = sorted([n for n, d in remaining.items() if not (d & set(remaining))])
        if not ready:
            raise RuntimeError(f"cycle in dep graph: {remaining!r}")
        for n in ready:
            order.append(n)
            del remaining[n]
    return order


def parse_script_order() -> list[str]:
    """Extract the ORDER=(...) list from publish.sh."""
    text = PUBLISH_SH.read_text()
    m = re.search(r"^ORDER=\((.*?)\)$", text, re.M | re.S)
    if not m:
        raise RuntimeError("could not find ORDER=(...) in publish.sh")
    return [line.strip() for line in m.group(1).splitlines() if line.strip() and not line.strip().startswith("#")]


def is_valid_order(order: list[str], deps: dict[str, set[str]]) -> tuple[bool, str]:
    """Every crate must appear after all of its solow-* deps."""
    position = {name: i for i, name in enumerate(order)}
    missing = set(deps) - set(order)
    if missing:
        return False, f"missing from script: {sorted(missing)}"
    for name, ds in deps.items():
        for dep in ds:
            if position[dep] >= position[name]:
                return False, f"{name} depends on {dep} but comes before it"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit non-zero on drift; don't print")
    args = parser.parse_args()

    deps = read_deps()
    expected = topo_sort(deps)
    actual = parse_script_order()

    ok, reason = is_valid_order(actual, deps)

    if args.check:
        if not ok:
            print(f"PUBLISH ORDER INVALID: {reason}", file=sys.stderr)
            return 1
        return 0

    print(f"Actual ORDER in publish.sh ({len(actual)} crates):")
    for i, n in enumerate(actual, 1):
        print(f"  {i:2d}. {n}")
    print()
    print(f"Topological sort of Cargo.toml deps ({len(expected)} crates):")
    for i, n in enumerate(expected, 1):
        print(f"  {i:2d}. {n}")
    print()
    if ok:
        print("OK — publish.sh order is a valid topological sort.")
        return 0
    print(f"MISMATCH — {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
