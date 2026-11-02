"""Guard benchmark constructor coverage and its subprocess CLI contract."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from medaugmentx.serialization import REGISTRY

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark.py"
if not SCRIPT.is_file():
    pytest.skip("benchmarks/ is absent in installed-wheel checks", allow_module_level=True)

spec = importlib.util.spec_from_file_location("mx_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("shape", [(16, 16), (8, 16, 16)])
def test_benchmark_measures_every_registered_leaf(shape):
    names = set(REGISTRY) - {"Compose", "OneOf", "SomeOf", "Guard"}
    if len(shape) == 2:
        names -= benchmark._REQUIRE_3D
    for name in sorted(names):
        result = benchmark.measure(name, shape, repeats=1)
        assert result["name"] == name
        assert len(result["latency_ms"]) == 1
        assert result["median_ms"] >= 0
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("shape", [(16, 16), (8, 16, 16)])
def test_spacing_benchmark_cli_writes_a_nonidentity_policy(tmp_path, shape):
    path = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--names",
            "ResampleToSpacing",
            "--shape",
            *map(str, shape),
            "--repeats",
            "1",
            "--json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(path.read_text())
    assert report["input"]["shape"] == list(shape)
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["name"] == "ResampleToSpacing"
    assert result["policy"]["params"]["spacing"] == (
        [1.25, 0.625, 0.625] if len(shape) == 3 else [0.625, 0.625]
    )
