"""Run and compare revision-bound FlashRWKV2 module benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(
    os.environ.get("FLASH_RWKV_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
RUNS = 3
REGRESSION_LIMIT = 0.002
BENCHMARKS = {
    "cmix/mix": ("benchmarks/cmix/mix/bench.py", "iters"),
    "cmix/sparse": ("benchmarks/cmix/sparse/bench.py", "samples"),
    "embedding": ("benchmarks/embedding/bench.py", "samples"),
    "head/l2wrap_ce": ("benchmarks/head/l2wrap_ce/bench.py", "samples"),
    "head/linear": ("benchmarks/head/linear/bench.py", "samples"),
    "loss/l2wrap_ce": ("benchmarks/loss/l2wrap_ce/bench.py", "samples"),
    "rl_infctx/wkv7": ("benchmarks/rl_infctx/wkv7/bench.py", "samples"),
    "tmix/a_gate": ("benchmarks/tmix/a_gate/bench.py", "iters"),
    "tmix/kk_a_gate": ("benchmarks/tmix/kk_a_gate/bench.py", "samples"),
    "tmix/kk_pre": ("benchmarks/tmix/kk_pre/bench.py", "samples"),
    "tmix/linear": ("benchmarks/tmix/linear/bench.py", "samples"),
    "tmix/lnx_rkvres_xg": ("benchmarks/tmix/lnx_rkvres_xg/bench.py", "samples"),
    "tmix/mix6": ("benchmarks/tmix/mix6/bench.py", "iters"),
    "tmix/normalization": ("benchmarks/tmix/normalization/bench.py", "samples"),
    "tmix/vres_gate": ("benchmarks/tmix/vres_gate/bench.py", "samples"),
    "tmix/wkv7": ("benchmarks/tmix/wkv7/bench.py", "wkv7"),
}


@dataclass(frozen=True)
class Metric:
    profile: str
    name: str
    unit: str
    direction: str
    summary: float
    raw_samples: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _environment(python: str) -> dict[str, Any]:
    program = r"""
import json, platform, torch
from pathlib import Path
extension = getattr(__import__('flashrwkv2'), '_C', None)
extension_path = Path(getattr(extension, '__file__', ''))
print(json.dumps({
  'python': platform.python_version(),
  'torch': torch.__version__,
  'torch_cuda': torch.version.cuda,
  'gpu': torch.cuda.get_device_name(0),
  'capability': list(torch.cuda.get_device_capability(0)),
  'extension_sha256': None,
  'extension_path': str(extension_path),
}))
"""
    payload = json.loads(
        subprocess.run(
            (python, "-c", program), check=True, capture_output=True, text=True
        ).stdout
    )
    extension_path = Path(payload["extension_path"])
    if not extension_path.is_file():
        raise SystemExit("flashrwkv2._C is not loaded from an installed wheel")
    payload["extension_sha256"] = _sha256(extension_path)
    payload["wheel_sha256"] = os.environ.get("FLASH_RWKV_WHEEL_SHA256")
    try:
        fields = "driver_version,temperature.gpu,power.draw,clocks.current.sm,clocks_event_reasons.active"
        output = subprocess.run(
            (
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        payload["nvidia_smi"] = output
        values = [value.strip() for value in output.split(",")]
        payload["driver"] = values[0] if values else None
        payload["throttle_reason"] = values[4] if len(values) > 4 else None
        if payload["throttle_reason"] not in {"Not Active", "Not Active."}:
            raise SystemExit(
                f"benchmark GPU reports an active clock event: {payload['throttle_reason']}"
            )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot record benchmark GPU environment: {error}") from error
    return payload


def _assert_gpu_idle() -> None:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
                "--id=0",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot verify exclusive benchmark GPU: {error}") from error
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if pids:
        raise SystemExit(f"benchmark GPU is not idle; active compute PIDs: {pids}")


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise SystemExit("benchmark did not emit a JSON object")


def _metrics(payload: dict[str, Any], module: str) -> list[Metric]:
    if module == "tmix/wkv7":
        metrics: list[Metric] = []
        for row in payload.get("results", []):
            correctness = row.get("correctness", {})
            if not correctness.get("passed") or "timing" not in row:
                raise SystemExit(f"WKV7 benchmark correctness/timing failed: {row}")
            timing = row["timing"]
            profile = "/".join(
                str(row[key]) for key in ("operator_shape", "case", "token_dtype")
            )
            raw = tuple(float(value) for value in timing["raw_latency_ms"])
            metrics.append(
                Metric(profile, "latency", "ms", "lower", float(timing["p50_ms"]), raw)
            )
        if not metrics:
            raise SystemExit("WKV7 benchmark emitted no measured profiles")
        return metrics

    correctness = payload.get("correctness", "passed")
    if correctness in {"failed", False, None}:
        raise SystemExit(f"benchmark correctness failed: {payload}")
    raw_value = payload.get("raw_latency_us", payload.get("latency_us"))
    if not isinstance(raw_value, list) or not raw_value:
        raise SystemExit(f"benchmark lacks raw latency samples: {payload}")
    raw = tuple(float(value) for value in raw_value)
    if any(not math.isfinite(value) or value <= 0 for value in raw):
        raise SystemExit(f"benchmark emitted invalid latency samples: {raw}")
    return [Metric("ci", "latency", "us", "lower", statistics.median(raw), raw)]


def _run_once(
    module: str, *, python: str, samples: int, run_index: int
) -> list[Metric]:
    script, sample_flag = BENCHMARKS[module]
    command = [python, str(ROOT / script)]
    output_path: Path | None = None
    if sample_flag == "wkv7":
        output_path = (
            Path(tempfile.gettempdir()) / f"flashrwkv2-{os.getpid()}-{run_index}.json"
        )
        command += [
            "--shapes",
            "h32d64",
            "--dtype",
            "bfloat16",
            "--warmup",
            "5",
            "--samples",
            str(samples),
            "--seed",
            "20260804",
            "--output",
            str(output_path),
        ]
    else:
        command += [f"--{sample_flag}", str(samples)]
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    payload = (
        json.loads(output_path.read_text())
        if output_path
        else _last_json(result.stdout)
    )
    if output_path:
        output_path.unlink(missing_ok=True)
    return _metrics(payload, module)


def run_benchmark(
    module: str, *, python: str, samples: int, revision: str
) -> dict[str, Any]:
    if module not in BENCHMARKS:
        raise SystemExit(f"no canonical benchmark configured for {module}")
    _assert_gpu_idle()
    environment = _environment(python)
    by_profile: dict[str, list[Metric]] = {}
    for run_index in range(RUNS):
        for metric in _run_once(
            module, python=python, samples=samples, run_index=run_index
        ):
            by_profile.setdefault(metric.profile, []).append(metric)
    profiles = []
    for profile, metrics in sorted(by_profile.items()):
        if len(metrics) != RUNS:
            raise SystemExit(
                f"{module}/{profile} has {len(metrics)} runs, expected {RUNS}"
            )
        identity = {(item.name, item.unit, item.direction) for item in metrics}
        if len(identity) != 1:
            raise SystemExit(f"metric identity changed across runs: {identity}")
        summaries = [item.summary for item in metrics]
        profiles.append(
            {
                "profile": profile,
                "metric": metrics[0].name,
                "unit": metrics[0].unit,
                "direction": metrics[0].direction,
                "runs": [
                    {"summary": item.summary, "raw_samples": item.raw_samples}
                    for item in metrics
                ],
                "mean": statistics.fmean(summaries),
            }
        )
    return {
        "schema_version": 1,
        "status": "candidate",
        "module": module,
        "target": "sm120",
        "revision": revision,
        "runtime_semantic_revision": _git(
            "log", "-1", "--format=%H", "--", "csrc", "flashrwkv2", "setup.py"
        ),
        "environment": environment,
        "independent_runs": RUNS,
        "samples_per_run": samples,
        "profiles": profiles,
    }


def _compatible(baseline: dict[str, Any], head: dict[str, Any]) -> None:
    for key in ("schema_version", "module", "target"):
        if baseline.get(key) != head.get(key):
            raise SystemExit(
                f"baseline {key} mismatch: {baseline.get(key)!r} != {head.get(key)!r}"
            )
    for key in ("gpu", "capability", "torch", "torch_cuda", "driver"):
        if baseline["environment"].get(key) != head["environment"].get(key):
            raise SystemExit(f"baseline environment mismatch for {key}")


def compare(baseline: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    _compatible(baseline, head)
    baseline_profiles = {row["profile"]: row for row in baseline["profiles"]}
    head_profiles = {row["profile"]: row for row in head["profiles"]}
    if baseline_profiles.keys() != head_profiles.keys():
        raise SystemExit("baseline/head profile sets differ")
    results = []
    passed = True
    for profile in sorted(head_profiles):
        base_row = baseline_profiles[profile]
        head_row = head_profiles[profile]
        for key in ("metric", "unit", "direction"):
            if base_row[key] != head_row[key]:
                raise SystemExit(f"metric {key} mismatch for {profile}")
        base_mean = float(base_row["mean"])
        head_mean = float(head_row["mean"])
        if not all(
            math.isfinite(value) and value > 0 for value in (base_mean, head_mean)
        ):
            raise SystemExit(f"non-finite/non-positive means for {profile}")
        if head_row["direction"] == "lower":
            regression = (head_mean - base_mean) / base_mean
        elif head_row["direction"] == "higher":
            regression = (base_mean - head_mean) / base_mean
        else:
            raise SystemExit(f"unknown metric direction: {head_row['direction']}")
        row_passed = regression < REGRESSION_LIMIT or math.isclose(
            regression, REGRESSION_LIMIT, rel_tol=0.0, abs_tol=1.0e-12
        )
        passed &= row_passed
        results.append(
            {
                "profile": profile,
                "baseline_mean": base_mean,
                "head_mean": head_mean,
                "regression": regression,
                "limit": REGRESSION_LIMIT,
                "passed": row_passed,
            }
        )
    return {
        "schema_version": 1,
        "module": head["module"],
        "baseline_revision": baseline["revision"],
        "head_revision": head["revision"],
        "passed": passed,
        "results": results,
    }


def _api_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def fetch_baseline(
    repository: str,
    module: str,
    output: Path,
    token: str,
    head: dict[str, Any],
) -> bool:
    safe_module = module.replace("/", "-")
    url = f"https://api.github.com/repos/{repository}/actions/artifacts?name=flashrwkv2-baseline-sm120-{safe_module}&per_page=100"
    payload = _api_json(url, token)
    candidates = sorted(
        (
            artifact
            for artifact in payload.get("artifacts", [])
            if not artifact.get("expired")
            and artifact.get("workflow_run", {}).get("head_branch") == "main"
        ),
        key=lambda artifact: artifact["created_at"],
        reverse=True,
    )
    if not candidates:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    for artifact in candidates:
        request = urllib.request.Request(
            artifact["archive_download_url"],
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
            },
        )
        with (
            urllib.request.urlopen(request) as response,
            tempfile.NamedTemporaryFile() as archive,
        ):
            archive.write(response.read())
            archive.flush()
            with zipfile.ZipFile(archive.name) as bundle:
                json_names = [
                    name for name in bundle.namelist() if name.endswith(".json")
                ]
                if len(json_names) != 1:
                    continue
                raw = bundle.read(json_names[0])
                payload = json.loads(raw)
                if payload.get("module") != module or payload.get("status") not in {
                    "approved",
                    "admin-approved",
                }:
                    continue
                try:
                    _compatible(payload, head)
                except SystemExit:
                    continue
                output.write_bytes(raw)
                return True
    return False


def approve(
    candidate: dict[str, Any], *, actor: str, reason: str, admin: bool
) -> dict[str, Any]:
    if (
        candidate.get("independent_runs") != RUNS
        or len(candidate.get("profiles", ())) == 0
    ):
        raise SystemExit("only a complete three-run candidate can become a baseline")
    approved = dict(candidate)
    approved["status"] = "admin-approved" if admin else "approved"
    approved["approval"] = {"actor": actor, "reason": reason}
    return approved


def _self_test() -> None:
    environment = {"gpu": "gpu", "capability": [12, 0], "torch": "x", "torch_cuda": "y"}

    def payload(value: float) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "module": "tmix/linear",
            "target": "sm120",
            "revision": str(value),
            "environment": environment,
            "profiles": [
                {
                    "profile": "ci",
                    "metric": "latency",
                    "unit": "us",
                    "direction": "lower",
                    "runs": [{"summary": value}] * 3,
                    "mean": value,
                }
            ],
        }

    assert compare(payload(100.0), payload(100.199))["passed"]
    assert compare(payload(100.0), payload(100.2))["passed"]
    assert not compare(payload(100.0), payload(100.2000001))["passed"]
    assert statistics.fmean([1.0, 2.0, 6.0]) == 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--module", required=True, choices=sorted(BENCHMARKS))
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--samples", type=int, default=30)
    run_parser.add_argument("--revision", default="")
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--head", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    fetch_parser = subparsers.add_parser("fetch-baseline")
    fetch_parser.add_argument("--repository", required=True)
    fetch_parser.add_argument("--module", required=True, choices=sorted(BENCHMARKS))
    fetch_parser.add_argument("--output", type=Path, required=True)
    fetch_parser.add_argument("--head", type=Path, required=True)
    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("--candidate", type=Path, required=True)
    approve_parser.add_argument("--output", type=Path, required=True)
    approve_parser.add_argument("--actor", required=True)
    approve_parser.add_argument("--reason", required=True)
    approve_parser.add_argument("--admin", action="store_true")
    subparsers.add_parser("self-test")
    args = parser.parse_args()

    if args.command == "self-test":
        _self_test()
        print("benchmark_gate self-test passed")
        return 0
    if args.command == "run":
        revision = args.revision or _git("rev-parse", "HEAD")
        payload = run_benchmark(
            args.module, python=args.python, samples=args.samples, revision=revision
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "compare":
        result = compare(
            json.loads(args.baseline.read_text()), json.loads(args.head.read_text())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, separators=(",", ":")))
        return 0 if result["passed"] else 1
    if args.command == "approve":
        payload = approve(
            json.loads(args.candidate.read_text()),
            actor=args.actor,
            reason=args.reason,
            admin=args.admin,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise SystemExit("GH_TOKEN is required to fetch a baseline artifact")
    found = fetch_baseline(
        args.repository,
        args.module,
        args.output,
        token,
        json.loads(args.head.read_text()),
    )
    print(json.dumps({"module": args.module, "found": found}))
    return 0 if found else 2


if __name__ == "__main__":
    raise SystemExit(main())
