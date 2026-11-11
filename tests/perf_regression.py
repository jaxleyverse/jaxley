# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Compare the performance of the working tree against a git ref. Local only.

    python tests/perf_regression.py --base main
    python tests/perf_regression.py --base HEAD --scenario cell net --repeats 5

Runs the scenarios of `tests/test_perf.py` with the `jaxley/` of the base ref (taken
with `git archive`, which does not touch your checkout) and with the `jaxley/` of the
working tree. Every run is a fresh process, base and head alternate, and the fastest
of `--repeats` runs counts. Prints the seconds of both and head / base, marking ratios
above 1.2 with `*`. Expect a few percent of noise.
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

import test_perf  # From this directory, which is on `sys.path` when run as a script.

REPO = Path(__file__).resolve().parent.parent


def worker(jaxley_root: str, out: str, names: list):
    """Times the scenarios `names` and writes the results to `out`."""
    import jaxley

    # Make sure that `PYTHONPATH` won over the installed jaxley.
    expected = Path(jaxley_root).resolve() / "jaxley"
    assert Path(jaxley.__file__).resolve().parent == expected, jaxley.__file__

    test_perf.warm_up()
    timings, errors = {}, {}
    for name in names:
        try:
            timings[name] = test_perf.measure(name)
        except Exception as e:  # E.g., the scenario uses an API the base lacks.
            errors[name] = f"{type(e).__name__}: {e}"
    Path(out).write_text(json.dumps({"timings": timings, "errors": errors}))


def run_worker(jaxley_root: Path, names: list, tmp: str) -> dict:
    """Runs `worker` in a fresh process that imports jaxley from `jaxley_root`."""
    out = Path(tmp) / "results.json"
    cmd = [sys.executable, __file__, "--worker", str(jaxley_root), str(out), *names]
    env = {**os.environ, "PYTHONPATH": str(jaxley_root)}
    proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Run with {jaxley_root} failed:\n{proc.stderr[-3000:]}")
    return json.loads(out.read_text())


def fastest(runs: list) -> dict:
    """Per scenario and column, the fastest time over all runs."""
    results = {}
    for name in runs[0]["timings"]:
        timings = [run["timings"][name] for run in runs]
        results[name] = {col: min(t[col] for t in timings) for col in timings[0]}
    return results


def main():
    if sys.argv[1:2] == ["--worker"]:
        return worker(sys.argv[2], sys.argv[3], sys.argv[4:])

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default="main", help="git ref (default: main)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per side")
    parser.add_argument(
        "--scenario", nargs="+", default=list(test_perf.SCENARIOS), help="names"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", args.base, "jaxley"],
            cwd=REPO,
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(Path(tmp) / "base", filter="data")

        roots = {"base": Path(tmp) / "base", "head": REPO}
        runs = {"base": [], "head": []}
        for i in range(args.repeats):
            for side in ("base", "head") if i % 2 == 0 else ("head", "base"):
                print(f"run {i + 1}/{args.repeats}: {side}", flush=True)
                runs[side].append(run_worker(roots[side], args.scenario, tmp))

    for side in ("base", "head"):
        for name, error in runs[side][0]["errors"].items():
            print(f"{name} failed on {side}: {error}")
    base, head = fastest(runs["base"]), fastest(runs["head"])
    ratios = {name: None for name in args.scenario}
    for name in set(base) & set(head):
        ratios[name] = {col: head[name][col] / base[name][col] for col in head[name]}

    print()
    print(test_perf.format_table(base, f"base [s] ({args.base})"))
    print()
    print(test_perf.format_table(head, "head [s] (working tree)"))
    print()
    print(test_perf.format_table(ratios, "head / base", "{:.2f}", mark_above=1.2))


if __name__ == "__main__":
    main()
