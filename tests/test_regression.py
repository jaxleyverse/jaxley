# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import hashlib
import json
import os
import time
from functools import wraps

import numpy as np
import pytest
from jax import jit

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import sparse_connect
from jaxley.synapses import IonotropicSynapse

# Every runtime test needs to have the following structure:
#
# @compare_to_baseline()
# def test_runtime_of_x(**kwargs) -> Dict:
#   t1 = time.time()
#   time.sleep(0.1)
#   # do something
#   t2 = time.time()
#   # do something else
#   t3 = time.time()
#   return {"sth": t2-t1, sth_else: t3-t2}

def load_json(fpath):
    dct = {}
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            dct = json.load(f)
    return dct


pytestmark = pytest.mark.regression  # mark all tests as regression tests in this file
NEW_BASELINE = os.environ["NEW_BASELINE"] if "NEW_BASELINE" in os.environ else 0
dirname = os.path.dirname(__file__)
fpath_baselines = os.path.join(dirname, "regression_test_baselines.json")
fpath_results = os.path.join(dirname, "regression_test_results.json")

tolerance = 0.2

baselines = load_json(fpath_baselines)
with open(fpath_results, "w") as f: # clear previous results
    f.write("{}")


def generate_regression_report(base_results, new_results):
    """Compare two sets of benchmark results and generate a diff report."""
    report = []
    for key in new_results:
        new_data = new_results[key]
        base_data = base_results.get(key)
        kwargs = ", ".join([f"{k}={v}" for k, v in new_data["input_kwargs"].items()])
        func_name = new_data["test_name"]
        func_signature = f"{func_name}({kwargs})"

        new_runtimes = new_data["runtimes"]
        base_runtimes = (
            {k: None for k in new_data.keys()}
            if base_data is None
            else base_data["runtimes"]
        )

        report.append(func_signature)
        for key, new_time in new_runtimes.items():
            base_time = base_runtimes.get(key)
            diff = None if base_time is None else ((new_time - base_time) / base_time)

            status = ""
            if diff is None:
                status = "🆕"
            elif diff > tolerance:
                status = "🔴"
            elif diff < 0:
                status = "🟢"
            else:
                status = "⚪"

            time_str = (
                f"({new_time:.3f}s)"
                if diff is None
                else f"({diff:+.2%} vs {base_time:.3f}s)"
            )
            report.append(f"{status} {key}: {time_str}.")
        report.append("")

    return "\n".join(report)


def generate_unique_key(d):
    # Generate a unique key for each test case. Makes it possible to compare tests
    # with different input_kwargs.
    hash_obj = hashlib.sha256(bytes(json.dumps(d, sort_keys=True), encoding="utf-8"))
    hash = hash_obj.hexdigest()
    return str(hash)


def append_to_json(fpath, test_name, input_kwargs, runtimes):
    header = {"test_name": test_name, "input_kwargs": input_kwargs}
    data = {generate_unique_key(header): {**header, "runtimes": runtimes}}

    # Save data to a JSON file
    result_data = load_json(fpath)
    result_data.update(data)

    with open(fpath, "w") as f:
        json.dump(result_data, f, indent=2)


class compare_to_baseline:
    def __init__(self, baseline_iters=3, test_iters=1):
        self.baseline_iters = baseline_iters
        self.test_iters = test_iters

    def __call__(self, func):
        @wraps(func)  # ensures kwargs exposed to pytest
        def test_wrapper(**kwargs):
            header = {"test_name": func.__name__, "input_kwargs": kwargs}
            key = generate_unique_key(header)

            runs = []
            num_iters = self.baseline_iters if NEW_BASELINE else self.test_iters
            for _ in range(num_iters):
                runtimes = func(**kwargs)
                runs.append(runtimes)
            runtimes = {k: np.mean([d[k] for d in runs]) for k in runs[0]}

            append_to_json(fpath_results, header["test_name"], header["input_kwargs"], runtimes)

            if not NEW_BASELINE:
                assert key in baselines, f"No basline found for {header}"
                func_baselines = baselines[key]["runtimes"]
                for key, baseline in func_baselines.items():
                    diff = (
                        float("nan")
                        if np.isclose(baseline, 0)
                        else (runtimes[key] - baseline) / baseline
                    )
                    assert runtimes[key] <= baseline * (
                        1 + tolerance
                    ), f"{key} is {diff:.2%} slower than the baseline."

        return test_wrapper


def build_net(num_cells, artificial=True, connect=True, connection_prob=0.0):
    _ = np.random.seed(1)  # For sparse connectivity matrix.

    if artificial:
        comp = jx.Compartment()
        branch = jx.Branch(comp, 2)
        depth = 3
        parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
        cell = jx.Cell(branch, parents=parents)
    else:
        dirname = os.path.dirname(__file__)
        fname = os.path.join(dirname, "swc_files", "morph.swc")
        cell = jx.read_swc(fname, nseg=4)
    net = jx.Network([cell for _ in range(num_cells)])

    # Channels.
    net.insert(HH())

    # Synapses.
    if connect:
        sparse_connect(
            net.cell("all"), net.cell("all"), IonotropicSynapse(), connection_prob
        )

    # Recordings.
    net[0, 1, 0].record(verbose=False)

    # Trainables.
    net.make_trainable("radius", verbose=False)
    params = net.get_parameters()

    net.to_jax()
    return net, params


@pytest.mark.parametrize(
    "num_cells, artificial, connect, connection_prob, voltage_solver",
    (
        # Test a single SWC cell with both solvers.
        pytest.param(1, False, False, 0.0, "jaxley.stone"),
        pytest.param(1, False, False, 0.0, "jax.sparse"),
        # Test a network of SWC cells with both solvers.
        pytest.param(10, False, True, 0.1, "jaxley.stone"),
        pytest.param(10, False, True, 0.1, "jax.sparse"),
        # Test a larger network of smaller neurons with both solvers.
        pytest.param(1000, True, True, 0.001, "jaxley.stone"),
        pytest.param(1000, True, True, 0.001, "jax.sparse"),
    ),
)
@compare_to_baseline(baseline_iters=3)
def test_runtime(
    num_cells: int,
    artificial: bool,
    connect: bool,
    connection_prob: float,
    voltage_solver: str,
):
    delta_t = 0.025
    t_max = 100.0

    def simulate(params):
        return jx.integrate(
            net,
            params=params,
            t_max=t_max,
            delta_t=delta_t,
            voltage_solver=voltage_solver,
        )

    runtimes = {}

    start_time = time.time()
    net, params = build_net(
        num_cells,
        artificial=artificial,
        connect=connect,
        connection_prob=connection_prob,
    )
    runtimes["build_time"] = time.time() - start_time

    jitted_simulate = jit(simulate)

    start_time = time.time()
    _ = jitted_simulate(params).block_until_ready()
    runtimes["compile_time"] = time.time() - start_time
    params[0]["radius"] = params[0]["radius"].at[0].set(0.5)

    start_time = time.time()
    _ = jitted_simulate(params).block_until_ready()
    runtimes["run_time"] = time.time() - start_time
    return runtimes  # @compare_to_baseline decorator will compare this to the baseline
