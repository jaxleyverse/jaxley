# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Compare the performance of the working tree against a git ref. Local only.

    python tests/perf_regression.py --base main
    python tests/perf_regression.py --base HEAD --scenario cell net --repeats 5

Runs `tests/test_perf.py` in fresh processes, once with the `jaxley/` of the base ref
(taken with `git archive`, which does not touch your checkout) and once with the
`jaxley/` of the working tree, `--repeats` times each. Prints the fastest seconds of
both and head / base. Expect ~10% of noise.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from test_perf import COLUMNS, SCENARIOS, format_table

TEST_PERF = Path(__file__).resolve().with_name("test_perf.py")
REPO = TEST_PERF.parent.parent


def run(jaxley_root, names: list) -> dict:
    """Times the scenarios `names` in a fresh process that imports `jaxley_root`."""
    env = {**os.environ, "PYTHONPATH": str(jaxley_root)}
    cmd = [sys.executable, str(TEST_PERF), *names]
    out = subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(out.stdout.splitlines()[-1])


def fastest(runs: list) -> dict:
    """Per scenario and column, the fastest of all runs."""
    return {
        name: {c: min(r[name][c] for r in runs) for c in COLUMNS} for name in runs[0]
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default="main", help="git ref (default: main)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per side")
    parser.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    args = parser.parse_args()

    runs = {"base": [], "head": []}
    with tempfile.TemporaryDirectory() as tmp:
        git_archive = ["git", "archive", args.base, "jaxley"]
        archive = subprocess.run(
            git_archive, cwd=REPO, check=True, stdout=subprocess.PIPE
        )
        tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(tmp, filter="data")

        roots = {"base": tmp, "head": REPO}
        for i in range(args.repeats):
            for side in runs:
                print(f"run {i + 1}/{args.repeats}: {side}", flush=True)
                runs[side].append(run(roots[side], args.scenario))

    base, head = fastest(runs["base"]), fastest(runs["head"])
    ratios = {
        name: {c: head[name][c] / base[name][c] for c in COLUMNS} for name in head
    }
    print()
    print(format_table(base, f"base [s] ({args.base})"))
    print()
    print(format_table(head, "head [s] (working tree)"))
    print()
    print(format_table(ratios, "head / base", fmt=".2f"))


if __name__ == "__main__":
    main()
