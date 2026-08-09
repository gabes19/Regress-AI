# RegressAI

RegressAI is a Python-first, browser-based regression analysis and stress-testing tool for students and researchers. Users upload a dataset, define a research question, choose a dependent variable, choose a main independent variable, add controls, and view regression results explained in a clean, AI-assisted, interactive dashboard.

## What Users Can Do

- Upload a CSV and inspect its columns instantly.
- Turn a research question into a regression setup.
- Pick dependent, main independent, and control variables.
- Run baseline and controlled OLS models in the browser.
- See how the main coefficient changes as controls are added.
- Review coefficients, p-values, and R-squared.
- Explore coefficient stability with an interactive Plotly chart.
- View bootstrap uncertainty and a bootstrap coefficient histogram.
- Offload expensive bootstrap workloads to CUDA-enabled GPU workers in the cloud
- Gain AI-generated insights into their regression results
- Exportable PDF/LaTeX Report

The goal is to help students, research assistants, and researchers quickly answer:

> Is this regression relationship stable, fragile, or misleading?

> How can I interpret the control variables in my model?

Expensive bootstrap workloads can be offloaded to CUDA-enabled GPU workers through RunPod Serverless.

## Cloud GPU execution

Cloud execution is optional and disabled by default. RegressAI prepares one finite numeric matrix locally, replaces user column names with neutral identifiers, compresses it, and sends only that matrix plus the model specification to a RunPod Serverless worker. Small analyses continue to use the existing statsmodels CPU implementation.

The initial worker target is an A4000 Flex worker with zero active workers and one maximum worker. Configure these environment variables in production:

```text
FLASK_SECRET_KEY=<long random secret>
GOOGLE_CLIENT_ID=<google oidc client id>
GOOGLE_CLIENT_SECRET=<google oidc client secret>
RUNPOD_ENABLED=true
RUNPOD_ENDPOINT_ID=<serverless endpoint id>
RUNPOD_API_KEY=<serverless api key>
```

The default policy routes jobs at 2,000 bootstrap iterations or 10,000,000 work units, limits each account to 3 GPU runs per day and 30 per month, and enforces global budgets of $2/day and $25/month. A 60-second A4000 Flex reservation is $0.012 at the configured $0.0002/second rate. Provider credentials, payloads, variable names, research questions, and results are never written to operational logs.

| Billable GPU execution | Estimated GPU cost | Including ≤$0.001 LLM allowance |
|---:|---:|---:|
| 5 seconds | $0.001 | ≤$0.002 |
| 10 seconds | $0.002 | ≤$0.003 |
| 30 seconds | $0.006 | ≤$0.007 |
| 60 seconds | $0.012 | ≤$0.013 |

At the limits above, one account's 30 full-duration monthly runs reserve at most about $0.36 of GPU execution. Flex scale-to-zero also avoids paying for a continuously active worker between jobs.

Build the pinned CUDA 12.9.2 worker from the repository root, then configure the RunPod endpoint for an A4000 Flex worker with zero active workers, one maximum worker, and a 60-second execution timeout:

```powershell
docker build -f gpu_worker/Dockerfile -t regressai-runpod-worker .
```

## CPU/GPU benchmarks

Normal analyses run on exactly one compute path. CPU/GPU comparisons are administrator-only and are never duplicated inside a user request.

Preview the benchmark matrix without making cloud requests:

```powershell
python benchmarks/benchmark_cpu_gpu.py
```

After deploying the worker and configuring credentials, explicitly authorize the live, billable campaign:

```powershell
python benchmarks/benchmark_cpu_gpu.py --confirm-live-run
```

The harness records median and p95 CPU time, GPU worker time, cold and warm end-to-end latency, RunPod delay/execution time, numerical parity, cost, and both compute-only and end-to-end speedup. It has a separate $1 campaign cap and writes auditable results to `benchmarks/results.json` and `benchmarks/results.csv`.

No live GPU benchmark has been recorded in this repository yet. The results table will be populated from those artifacts after the first controlled RunPod run:

<!-- BENCHMARK_RESULTS_START -->
| Workload | CPU median | GPU compute median | GPU warm/cold E2E | Compute speedup | E2E speedup | GPU cost | Parity |
|---|---:|---:|---:|---:|---:|---:|---|
| Pending live benchmark | — | — | — | — | — | — | — |
<!-- BENCHMARK_RESULTS_END -->

The recommended production crossover is the smallest passing workload with at least 1.25× median warm end-to-end speedup. The harness records that value as `recommended_gpu_min_work_units`; set `GPU_MIN_WORK_UNITS` to it after reviewing the raw results. If none qualifies, leave `RUNPOD_ENABLED=false`.

![alt text](image.png)
