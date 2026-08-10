"""Administrator-only CPU/RunPod benchmark harness.

Dry-run is the default. A live cloud request requires --confirm-live-run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from config import Config
from regressionlab.services.bootstrap_cpu import bootstrap_coefficient
from regressionlab.services.data_processing import prepare_analysis_data
from regressionlab.services.gpu_client import RunPodClient, calculate_gpu_cost
from regressionlab.services.gpu_contract import build_gpu_request, merge_gpu_model_metadata
from regressionlab.services.gpu_usage import (
    initialize_gpu_database,
    reconcile_gpu_run,
    reserve_benchmark_run,
)
from regressionlab.services.regression import fit_models


PROFILES = [
    ("Small A", 1_000, 3, 500, 5),
    ("Small B", 1_000, 3, 2_000, 5),
    ("Medium A", 10_000, 5, 1_000, 5),
    ("Medium B", 10_000, 5, 5_000, 5),
    ("Large A", 50_000, 10, 2_000, 3),
    ("Large B", 50_000, 10, 5_000, 3),
    ("Extra large", 100_000, 10, 10_000, 3),
]
PROFILE_NAMES = tuple(profile[0] for profile in PROFILES)


def select_profiles(requested_names):
    """Return requested profiles in canonical campaign order."""
    if not requested_names:
        return list(PROFILES)
    unknown = sorted(set(requested_names) - set(PROFILE_NAMES))
    if unknown:
        raise ValueError(f"Unknown benchmark profile(s): {', '.join(unknown)}")
    requested = set(requested_names)
    return [profile for profile in PROFILES if profile[0] in requested]


def save_checkpoint(output_dir, raw_records, completed_profiles, campaign_cost):
    """Atomically persist resumable state without credentials or analysis payloads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    temporary_path = output_dir / "checkpoint.json.tmp"
    completed = set(completed_profiles)
    document = {
        "schema_version": 1,
        "campaign_cost_usd": str(campaign_cost),
        "completed_profiles": [name for name in PROFILE_NAMES if name in completed],
        "raw_runs": raw_records,
    }
    temporary_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary_path.replace(checkpoint_path)


def load_checkpoint(output_dir):
    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        raise ValueError(f"Benchmark checkpoint does not exist: {checkpoint_path}")
    document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark checkpoint schema version.")
    completed_profiles = set(document.get("completed_profiles", []))
    if completed_profiles - set(PROFILE_NAMES):
        raise ValueError("Benchmark checkpoint contains unknown profiles.")
    campaign_cost = Decimal(document.get("campaign_cost_usd", "0"))
    if campaign_cost < 0:
        raise ValueError("Benchmark checkpoint contains a negative campaign cost.")
    raw_records = document.get("raw_runs", [])
    if not isinstance(raw_records, list):
        raise ValueError("Benchmark checkpoint raw_runs must be a list.")
    return raw_records, completed_profiles, campaign_cost


def percentile(values, percentile_value):
    return float(np.percentile(np.asarray(values, dtype=float), percentile_value))


def describe_speed(cpu_seconds, gpu_seconds):
    speedup = cpu_seconds / gpu_seconds
    improvement = (1 - gpu_seconds / cpu_seconds) * 100
    label = f"{improvement:.1f}% faster" if improvement >= 0 else f"{abs(improvement):.1f}% slower"
    return speedup, improvement, label


def synthetic_data(rows, predictor_count, seed=20260809):
    rng = np.random.default_rng(seed)
    values = {f"p{i}": rng.normal(size=rows) for i in range(predictor_count)}
    noise = rng.normal(scale=2.0, size=rows)
    values["outcome"] = 1.2 * values["p0"] + sum(
        (0.15 / (i + 1)) * values[f"p{i}"] for i in range(1, predictor_count)
    ) + noise
    frame = pd.DataFrame(values)
    controls = [f"p{i}" for i in range(1, predictor_count)]
    prepared = prepare_analysis_data(frame, "outcome", "p0", controls)
    return prepared, controls


def explicit_indices(rows, iterations, seed=42):
    if not uses_explicit_indices(rows, iterations):
        return None
    return np.random.default_rng(seed).integers(0, rows, size=(iterations, rows)).tolist()


def uses_explicit_indices(rows, iterations):
    """Exact parity payloads are intentionally limited and are not production-like."""
    return rows * iterations <= 2_000_000


def run_cpu(prepared, controls, iterations, indices):
    started = perf_counter()
    models = fit_models(prepared, "outcome", "p0", controls)
    bootstrap = bootstrap_coefficient(
        prepared, "p0", iterations, random_seed=42, bootstrap_indices=indices
    )
    return perf_counter() - started, models, bootstrap


def run_gpu(client, request_data, rows, predictors, iterations):
    reservation = reserve_benchmark_run(
        Config.GPU_USAGE_DATABASE,
        rows * (predictors + 1) * iterations,
        iterations,
        Config.GPU_RESERVED_COST_USD,
        Config.GPU_GLOBAL_DAILY_BUDGET_USD,
        Config.GPU_GLOBAL_MONTHLY_BUDGET_USD,
        Config.GPU_GLOBAL_MAX_IN_FLIGHT,
    )
    try:
        result = client.run(request_data)
        reconcile_gpu_run(
            Config.GPU_USAGE_DATABASE, reservation.run_id, "completed",
            result.execution_time_ms, result.job_id, Config.GPU_PRICE_PER_SECOND_USD,
        )
        return result
    except Exception:
        reconcile_gpu_run(
            Config.GPU_USAGE_DATABASE, reservation.run_id, "failed",
            price_per_second=Config.GPU_PRICE_PER_SECOND_USD,
        )
        raise


def parity_ok(cpu_models, cpu_bootstrap, gpu_output, exact_samples):
    gpu_models = merge_gpu_model_metadata(gpu_output, "outcome", "p0", [
        f"p{i}" for i in range(1, len(cpu_models))
    ])
    metric_names = ["coefficient", "standard_error", "t_value", "p_value", "r_squared", "adjusted_r_squared", "rmse"]
    for cpu_model, gpu_model in zip(cpu_models, gpu_models):
        for metric in metric_names:
            if not np.isclose(cpu_model[metric], gpu_model[metric], rtol=1e-5, atol=1e-7, equal_nan=True):
                return False
    gpu_bootstrap = gpu_output.bootstrap.model_dump(mode="json")
    if exact_samples:
        return np.allclose(cpu_bootstrap["samples"], gpu_bootstrap["samples"], rtol=1e-6, atol=1e-7)
    return all(
        np.isclose(cpu_bootstrap[key], gpu_bootstrap[key], rtol=0.08, atol=1e-5)
        for key in ("mean", "standard_error")
    )


def aggregate(profile, records):
    measured = [record for record in records if not record["cold"]]
    cpu = [record["cpu_seconds"] for record in measured]
    worker = [record["gpu_worker_seconds"] for record in measured]
    warm_e2e = [record["gpu_end_to_end_seconds"] for record in records if not record["cold"]]
    cold_e2e = [record["gpu_end_to_end_seconds"] for record in records if record["cold"]]
    cpu_median = statistics.median(cpu)
    worker_median = statistics.median(worker)
    e2e_median = statistics.median(warm_e2e)
    compute_speedup, _, _ = describe_speed(cpu_median, worker_median)
    e2e_speedup, improvement, label = describe_speed(cpu_median, e2e_median)
    execution_costs = [Decimal(record["gpu_cost_usd"]) for record in records]
    return {
        "profile": profile[0], "rows": profile[1], "predictors": profile[2],
        "bootstrap_iterations": profile[3], "repetitions": profile[4],
        "cpu_median_seconds": cpu_median, "cpu_p95_seconds": percentile(cpu, 95),
        "gpu_compute_median_seconds": worker_median,
        "gpu_warm_e2e_median_seconds": e2e_median,
        "gpu_cold_e2e_seconds": statistics.median(cold_e2e) if cold_e2e else None,
        "gpu_e2e_p95_seconds": percentile(warm_e2e, 95),
        "compute_speedup": compute_speedup, "end_to_end_speedup": e2e_speedup,
        "percentage_improvement": improvement, "comparison": label,
        "median_gpu_cost_usd": str(statistics.median(execution_costs)),
        "cost_per_1000_bootstraps_usd": str(statistics.median(execution_costs) * Decimal(1000) / Decimal(profile[3])),
        "parity_passed": all(record["parity_passed"] for record in records),
        "routing_eligible": not uses_explicit_indices(profile[1], profile[3]),
    }


def aggregate_completed_profiles(raw_records, completed_profiles):
    results = []
    for profile in PROFILES:
        name = profile[0]
        if name not in completed_profiles:
            continue
        records = [record for record in raw_records if record["profile"] == name]
        results.append(aggregate(profile, records))
    return results


def recommend_gpu_min_work_units(results):
    eligible = [
        item for item in results
        if item["routing_eligible"]
        and item["parity_passed"]
        and item["end_to_end_speedup"] >= 1.25
    ]
    return min(
        (
            item["rows"]
            * (item["predictors"] + 1)
            * item["bootstrap_iterations"]
            for item in eligible
        ),
        default=None,
    )


def write_results(results, raw_records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendation = recommend_gpu_min_work_units(results)
    document = {"pricing_usd_per_second": "0.0002", "recommended_gpu_min_work_units": recommendation, "profiles": results, "raw_runs": raw_records}
    (output_dir / "results.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    update_readme_table(results)


def update_readme_table(results):
    readme_path = PROJECT_ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    start_marker = "<!-- BENCHMARK_RESULTS_START -->"
    end_marker = "<!-- BENCHMARK_RESULTS_END -->"
    if start_marker not in text or end_marker not in text:
        raise ValueError("README benchmark markers are missing.")
    rows = [
        "| Workload | CPU median | GPU compute median | GPU warm/cold E2E | Compute speedup | E2E speedup | GPU cost | Parity | Routing evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in results:
        cold = item["gpu_cold_e2e_seconds"]
        cold_text = f"{cold:.3f}s" if cold is not None else "—"
        rows.append(
            f"| {item['profile']} | {item['cpu_median_seconds']:.3f}s | "
            f"{item['gpu_compute_median_seconds']:.3f}s | "
            f"{item['gpu_warm_e2e_median_seconds']:.3f}s / {cold_text} | "
            f"{item['compute_speedup']:.2f}× | {item['end_to_end_speedup']:.2f}× "
            f"({item['comparison']}) | ${item['median_gpu_cost_usd']} | "
            f"{'pass' if item['parity_passed'] else 'FAIL'} | "
            f"{'eligible' if item['routing_eligible'] else 'parity-only'} |"
        )
    replacement = start_marker + "\n" + "\n".join(rows) + "\n" + end_marker
    before, remainder = text.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    readme_path.write_text(before + replacement + after, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-live-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks")
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_NAMES,
        help="Run only this profile; repeat this option to select multiple profiles.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from output-dir/checkpoint.json.",
    )
    args = parser.parse_args()
    selected_profiles = select_profiles(args.profile)
    print("profile,rows,predictors,iterations,repetitions")
    for profile in selected_profiles:
        print(",".join(map(str, profile)))
    if not args.confirm_live_run:
        print("Dry run only. Add --confirm-live-run to submit billable RunPod jobs.")
        return 0
    client = RunPodClient(
        Config.RUNPOD_ENDPOINT_ID,
        Config.RUNPOD_API_KEY,
        price_per_second=Config.GPU_PRICE_PER_SECOND_USD,
    )
    if not Config.RUNPOD_ENABLED or not client.configured:
        raise SystemExit("RUNPOD_ENABLED=true, RUNPOD_ENDPOINT_ID, and RUNPOD_API_KEY are required.")

    initialize_gpu_database(Config.GPU_USAGE_DATABASE)
    if args.resume:
        try:
            raw_records, completed_profiles, campaign_cost = load_checkpoint(args.output_dir)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(
            f"Resuming {len(completed_profiles)} completed profile(s); "
            f"recorded campaign cost ${campaign_cost}."
        )
    else:
        raw_records = []
        completed_profiles = set()
        campaign_cost = Decimal("0")
    first_request = not any(record.get("cold") for record in raw_records)
    for profile in selected_profiles:
        name, rows, predictors, iterations, repetitions = profile
        if name in completed_profiles:
            print(f"Skipping completed profile: {name}")
            continue
        prepared, controls = synthetic_data(rows, predictors)
        indices = explicit_indices(rows, iterations)
        request_data = build_gpu_request(prepared, "p0", controls, iterations, bootstrap_indices=indices)
        # One unrecorded CPU/GPU warm-up per profile. The first GPU request is also retained as cold-start evidence.
        _, warm_cpu_models, warm_cpu_bootstrap = run_cpu(prepared, controls, iterations, indices)
        if campaign_cost + Config.GPU_RESERVED_COST_USD > Config.BENCHMARK_CAMPAIGN_BUDGET_USD:
            raise SystemExit("Benchmark campaign budget would be exceeded.")
        cold_result = run_gpu(client, request_data, rows, predictors, iterations)
        cold_cost = calculate_gpu_cost(cold_result.execution_time_ms, Config.GPU_PRICE_PER_SECOND_USD) or Config.GPU_RESERVED_COST_USD
        campaign_cost += cold_cost
        cold_record = {
            "profile": name, "repetition": 0, "cold": first_request,
            "cpu_seconds": _, "gpu_worker_seconds": cold_result.output.runtime_seconds,
            "gpu_end_to_end_seconds": cold_result.end_to_end_seconds,
            "delay_time_ms": cold_result.delay_time_ms, "execution_time_ms": cold_result.execution_time_ms,
            "encoded_payload_bytes": len(request_data.matrix_payload.encode("ascii")),
            "gpu_name": cold_result.output.gpu_name, "gpu_cost_usd": str(cold_cost),
            "parity_passed": parity_ok(warm_cpu_models, warm_cpu_bootstrap, cold_result.output, indices is not None),
        }
        if first_request:
            raw_records.append(cold_record)
        first_request = False
        save_checkpoint(args.output_dir, raw_records, completed_profiles, campaign_cost)
        completed_repetitions = {
            int(record["repetition"])
            for record in raw_records
            if record["profile"] == name and not record["cold"]
        }
        for repetition in range(1, repetitions + 1):
            if repetition in completed_repetitions:
                print(f"Skipping completed measurement: {name} repetition {repetition}")
                continue
            if campaign_cost + Config.GPU_RESERVED_COST_USD > Config.BENCHMARK_CAMPAIGN_BUDGET_USD:
                raise SystemExit("Benchmark campaign budget would be exceeded.")
            cpu_seconds, cpu_models, cpu_bootstrap = run_cpu(prepared, controls, iterations, indices)
            gpu_result = run_gpu(client, request_data, rows, predictors, iterations)
            cost = calculate_gpu_cost(gpu_result.execution_time_ms, Config.GPU_PRICE_PER_SECOND_USD) or Config.GPU_RESERVED_COST_USD
            campaign_cost += cost
            raw_records.append({
                "profile": name, "repetition": repetition, "cold": False,
                "cpu_seconds": cpu_seconds, "gpu_worker_seconds": gpu_result.output.runtime_seconds,
                "gpu_end_to_end_seconds": gpu_result.end_to_end_seconds,
                "delay_time_ms": gpu_result.delay_time_ms, "execution_time_ms": gpu_result.execution_time_ms,
                "encoded_payload_bytes": len(request_data.matrix_payload.encode("ascii")),
                "gpu_name": gpu_result.output.gpu_name, "gpu_cost_usd": str(cost),
                "parity_passed": parity_ok(cpu_models, cpu_bootstrap, gpu_result.output, indices is not None),
            })
            save_checkpoint(args.output_dir, raw_records, completed_profiles, campaign_cost)
        completed_profiles.add(name)
        save_checkpoint(args.output_dir, raw_records, completed_profiles, campaign_cost)
        aggregates = aggregate_completed_profiles(raw_records, completed_profiles)
        write_results(aggregates, raw_records, args.output_dir)
        print(f"Completed profile: {name}")
    aggregates = aggregate_completed_profiles(raw_records, completed_profiles)
    if aggregates:
        write_results(aggregates, raw_records, args.output_dir)
    print(f"Wrote benchmark results. Estimated GPU cost: ${campaign_cost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
