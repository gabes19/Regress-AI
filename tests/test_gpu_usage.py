from datetime import datetime, timezone
from decimal import Decimal

import pytest

from regressionlab.services.gpu_client import calculate_gpu_cost
from regressionlab.services.gpu_usage import (
    GPUQuotaError,
    initialize_gpu_database,
    reconcile_gpu_run,
    remaining_gpu_quota,
    reserve_benchmark_run,
    reserve_gpu_run,
    upsert_google_user,
)


def user_and_database(tmp_path):
    database = tmp_path / "usage.sqlite"
    initialize_gpu_database(database)
    user = upsert_google_user(database, {
        "sub": "google-123", "email": "user@example.com",
        "email_verified": True, "name": "Researcher",
    })
    return user, database


def reserve(database, user_id, now=None):
    return reserve_gpu_run(
        database, user_id, 10_000_000, 2_000, Decimal("0.012"),
        3, 30, Decimal("2"), Decimal("25"), now=now,
    )


def test_gpu_cost_rounds_to_billable_seconds():
    assert calculate_gpu_cost(0) == Decimal("0.0000")
    assert calculate_gpu_cost(1) == Decimal("0.0002")
    assert calculate_gpu_cost(1_001) == Decimal("0.0004")
    assert calculate_gpu_cost(None) is None


def test_reservation_is_atomic_and_enforces_inflight_and_daily_quota(tmp_path):
    user, database = user_and_database(tmp_path)
    now = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    first = reserve(database, user["id"], now)
    with pytest.raises(GPUQuotaError, match="already"):
        reserve(database, user["id"], now)
    reconcile_gpu_run(database, first.run_id, "completed", execution_time_ms=1_001)

    second = reserve(database, user["id"], now)
    reconcile_gpu_run(database, second.run_id, "failed")
    third = reserve(database, user["id"], now)
    reconcile_gpu_run(database, third.run_id, "completed", execution_time_ms=500)
    with pytest.raises(GPUQuotaError, match="Daily"):
        reserve(database, user["id"], now)
    assert remaining_gpu_quota(database, user["id"], 3, 30, now) == {
        "daily": 0, "monthly": 27
    }


def test_verified_google_email_is_required(tmp_path):
    database = tmp_path / "usage.sqlite"
    initialize_gpu_database(database)
    with pytest.raises(ValueError, match="verified"):
        upsert_google_user(database, {
            "sub": "1", "email": "user@example.com", "email_verified": False
        })


def test_benchmark_reservation_shares_global_concurrency_and_budget(tmp_path):
    user, database = user_and_database(tmp_path)
    benchmark = reserve_benchmark_run(
        database, 10_000, 500, Decimal("0.012"), Decimal("2"), Decimal("25")
    )
    with pytest.raises(GPUQuotaError, match="busy"):
        reserve(database, user["id"])
    reconcile_gpu_run(database, benchmark.run_id, "completed", execution_time_ms=500)
    assert reserve(database, user["id"]).run_id > benchmark.run_id
