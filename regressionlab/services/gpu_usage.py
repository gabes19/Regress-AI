"""SQLite-backed GPU identity, quota, concurrency, and budget accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3

from .gpu_client import calculate_gpu_cost


class GPUQuotaError(Exception):
    """A recoverable local policy rejection; no provider request was made."""


@dataclass(frozen=True)
class GPUReservation:
    run_id: int
    remaining_daily: int
    remaining_monthly: int


def _connect(database_path):
    connection = sqlite3.connect(Path(database_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_gpu_database(database_path):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_sub TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                display_name TEXT,
                avatar_url TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gpu_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                provider_job_id TEXT,
                status TEXT NOT NULL,
                workload_units INTEGER NOT NULL,
                bootstrap_iterations INTEGER NOT NULL,
                reserved_cost_usd TEXT NOT NULL,
                actual_cost_usd TEXT,
                execution_time_ms INTEGER,
                is_benchmark INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_gpu_runs_user_created
                ON gpu_runs(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_gpu_runs_status
                ON gpu_runs(status);
        """)


def upsert_google_user(database_path, claims):
    if not claims.get("sub") or not claims.get("email") or claims.get("email_verified") is not True:
        raise ValueError("A verified Google account is required.")
    now = datetime.now(timezone.utc).isoformat()
    with _connect(database_path) as connection:
        connection.execute(
            """INSERT INTO users
               (google_sub, email, display_name, avatar_url, created_at, last_login_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(google_sub) DO UPDATE SET
                   email=excluded.email,
                   display_name=excluded.display_name,
                   avatar_url=excluded.avatar_url,
                   last_login_at=excluded.last_login_at""",
            (claims["sub"], claims["email"], claims.get("name"), claims.get("picture"), now, now),
        )
        user = connection.execute(
            "SELECT id, email, display_name, avatar_url, status FROM users WHERE google_sub = ?",
            (claims["sub"],),
        ).fetchone()
    if user["status"] != "active":
        raise ValueError("This account is disabled.")
    return dict(user)


def _period_starts(now):
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month = day.replace(day=1)
    return day.isoformat(), month.isoformat()


def _effective_spend(connection, since):
    rows = connection.execute(
        "SELECT reserved_cost_usd, actual_cost_usd, status FROM gpu_runs WHERE created_at >= ?",
        (since,),
    ).fetchall()
    return sum(
        (
            Decimal(row["actual_cost_usd"])
            if row["actual_cost_usd"] is not None
            else Decimal(row["reserved_cost_usd"])
        )
        for row in rows
    )


def _expire_stale_reservations(connection, now, stale_after_minutes=10):
    threshold = (now - timedelta(minutes=stale_after_minutes)).isoformat()
    connection.execute(
        """UPDATE gpu_runs
           SET status='timed_out', actual_cost_usd=reserved_cost_usd, completed_at=?
           WHERE status='in_flight' AND created_at < ?""",
        (now.isoformat(), threshold),
    )


def reserve_gpu_run(
    database_path,
    user_id,
    workload_units,
    bootstrap_iterations,
    reserved_cost,
    daily_user_limit,
    monthly_user_limit,
    global_daily_budget,
    global_monthly_budget,
    global_max_in_flight=1,
    now=None,
):
    now = now or datetime.now(timezone.utc)
    day_start, month_start = _period_starts(now)
    reserved_cost = Decimal(reserved_cost)
    connection = _connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _expire_stale_reservations(connection, now)
        user = connection.execute(
            "SELECT status FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user is None or user["status"] != "active":
            raise GPUQuotaError("Sign in with an active account to use cloud GPU.")

        inflight_user = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE user_id = ? AND status = 'in_flight'",
            (user_id,),
        ).fetchone()[0]
        inflight_global = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE status = 'in_flight'"
        ).fetchone()[0]
        if inflight_user:
            raise GPUQuotaError("This account already has a GPU analysis running.")
        if inflight_global >= global_max_in_flight:
            raise GPUQuotaError("The GPU worker is currently busy.")

        daily_count = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE user_id = ? AND is_benchmark = 0 AND created_at >= ?",
            (user_id, day_start),
        ).fetchone()[0]
        monthly_count = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE user_id = ? AND is_benchmark = 0 AND created_at >= ?",
            (user_id, month_start),
        ).fetchone()[0]
        if daily_count >= daily_user_limit:
            raise GPUQuotaError("Daily GPU run limit reached.")
        if monthly_count >= monthly_user_limit:
            raise GPUQuotaError("Monthly GPU run limit reached.")
        if _effective_spend(connection, day_start) + reserved_cost > Decimal(global_daily_budget):
            raise GPUQuotaError("The daily cloud GPU budget is currently exhausted.")
        if _effective_spend(connection, month_start) + reserved_cost > Decimal(global_monthly_budget):
            raise GPUQuotaError("The monthly cloud GPU budget is currently exhausted.")

        cursor = connection.execute(
            """INSERT INTO gpu_runs
               (user_id, status, workload_units, bootstrap_iterations,
                reserved_cost_usd, created_at)
               VALUES (?, 'in_flight', ?, ?, ?, ?)""",
            (user_id, workload_units, bootstrap_iterations, str(reserved_cost), now.isoformat()),
        )
        connection.commit()
        return GPUReservation(
            run_id=cursor.lastrowid,
            remaining_daily=max(0, daily_user_limit - daily_count - 1),
            remaining_monthly=max(0, monthly_user_limit - monthly_count - 1),
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reserve_benchmark_run(
    database_path,
    workload_units,
    bootstrap_iterations,
    reserved_cost,
    global_daily_budget,
    global_monthly_budget,
    global_max_in_flight=1,
    now=None,
):
    """Reserve an admin benchmark without consuming an account quota."""
    now = now or datetime.now(timezone.utc)
    day_start, month_start = _period_starts(now)
    reserved_cost = Decimal(reserved_cost)
    connection = _connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _expire_stale_reservations(connection, now)
        in_flight = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE status='in_flight'"
        ).fetchone()[0]
        if in_flight >= global_max_in_flight:
            raise GPUQuotaError("The GPU worker is currently busy.")
        if _effective_spend(connection, day_start) + reserved_cost > Decimal(global_daily_budget):
            raise GPUQuotaError("The daily cloud GPU budget is currently exhausted.")
        if _effective_spend(connection, month_start) + reserved_cost > Decimal(global_monthly_budget):
            raise GPUQuotaError("The monthly cloud GPU budget is currently exhausted.")
        cursor = connection.execute(
            """INSERT INTO gpu_runs
               (user_id, status, workload_units, bootstrap_iterations,
                reserved_cost_usd, is_benchmark, created_at)
               VALUES (NULL, 'in_flight', ?, ?, ?, 1, ?)""",
            (workload_units, bootstrap_iterations, str(reserved_cost), now.isoformat()),
        )
        connection.commit()
        return GPUReservation(cursor.lastrowid, 0, 0)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reconcile_gpu_run(
    database_path,
    run_id,
    status,
    execution_time_ms=None,
    provider_job_id=None,
    price_per_second=Decimal("0.0002"),
):
    if status not in {"completed", "failed", "timed_out"}:
        raise ValueError("Invalid terminal GPU run status.")
    now = datetime.now(timezone.utc).isoformat()
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT reserved_cost_usd FROM gpu_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError("GPU run reservation was not found.")
        calculated = calculate_gpu_cost(execution_time_ms, price_per_second)
        actual_cost = calculated if calculated is not None else Decimal(row["reserved_cost_usd"])
        connection.execute(
            """UPDATE gpu_runs SET status=?, provider_job_id=?, execution_time_ms=?,
               actual_cost_usd=?, completed_at=? WHERE id=? AND status='in_flight'""",
            (status, provider_job_id, execution_time_ms, str(actual_cost), now, run_id),
        )


def remaining_gpu_quota(database_path, user_id, daily_limit, monthly_limit, now=None):
    now = now or datetime.now(timezone.utc)
    day_start, month_start = _period_starts(now)
    with _connect(database_path) as connection:
        daily = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE user_id=? AND is_benchmark=0 AND created_at>=?",
            (user_id, day_start),
        ).fetchone()[0]
        monthly = connection.execute(
            "SELECT COUNT(*) FROM gpu_runs WHERE user_id=? AND is_benchmark=0 AND created_at>=?",
            (user_id, month_start),
        ).fetchone()[0]
    return {
        "daily": max(0, daily_limit - daily),
        "monthly": max(0, monthly_limit - monthly),
    }
