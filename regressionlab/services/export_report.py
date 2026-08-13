# Handles LaTeX report generation and exporting.
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
import zipfile
import subprocess
import matplotlib
import json
import uuid
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regressionlab.services.regression import clean_metric
from regressionlab.services.llm_summary import render_analysis_summary_text


class ReportExportError(Exception):
    """Base exception for report export failures."""


class ExportNotFoundError(ReportExportError):
    """Raised when an export token is invalid or has no stored payload."""


class ReportGenerationError(ReportExportError):
    """Raised when report artifacts cannot be generated."""


OWNER_KEY_PATTERN = re.compile(r"(?:user:[1-9][0-9]*|guest:[0-9a-f]{32})")


def normalize_owner_key(owner_id):
    """Normalize current owner keys and legacy numeric user identifiers."""
    if owner_id is None:
        return None
    if isinstance(owner_id, bool):
        raise ExportNotFoundError("Export payload is invalid.")
    if isinstance(owner_id, int) or str(owner_id).isdigit():
        owner_key = f"user:{int(owner_id)}"
    else:
        owner_key = str(owner_id)
    if not OWNER_KEY_PATTERN.fullmatch(owner_key):
        raise ExportNotFoundError("Export payload is invalid.")
    return owner_key

def build_export_payload(
    research_question,
    dependent_variable,
    main_independent_variable,
    controls,
    bootstrap_iterations,
    model_results,
    baseline_coefficient,
    final_coefficient,
    coefficient_change,
    coefficient_chart,
    bootstrap_results,
    llm_summary,
    compute_mode="CPU",
    runtime_seconds=None,
):
    '''Build a compact report payload for immediate PDF/LaTeX export.'''
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_question": research_question,
        "dependent_variable": dependent_variable,
        "main_independent_variable": main_independent_variable,
        "controls": controls,
        "bootstrap_iterations": bootstrap_iterations,
        "models": model_results,
        "baseline_coefficient": baseline_coefficient,
        "final_coefficient": final_coefficient,
        "coefficient_change": coefficient_change,
        "coefficient_chart": coefficient_chart,
        "bootstrap_results": bootstrap_results,
        "llm_summary": llm_summary,
        "compute_mode": compute_mode,
        "runtime_seconds": runtime_seconds,
    }

def store_export_payload(export_payload, reports_folder, owner_id=None):
    '''Store the current analysis payload for export downloads.'''
    reports_folder = Path(reports_folder)
    export_token = uuid.uuid4().hex
    payload_dir = reports_folder / "export_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    payload_path = payload_dir / f"{export_token}.json"
    stored_payload = dict(export_payload)
    stored_payload["owner_id"] = normalize_owner_key(owner_id)
    with payload_path.open("w", encoding="utf-8") as file:
        json.dump(stored_payload, file, indent=2)

    return export_token

def validate_export_token(export_token):
    if not re.fullmatch(r"[0-9a-f]{32}", export_token or ""):
        raise ExportNotFoundError("Invalid export token.")

def load_export_payload(
    export_token,
    reports_folder,
    owner_id=None,
    accepted_owner_ids=None,
    enforce_owner=False,
):
    validate_export_token(export_token)
    payload_path = (
        Path(reports_folder)
        / "export_payloads"
        / f"{export_token}.json"
    )

    if not payload_path.exists():
        raise ExportNotFoundError("Export payload not found.")

    with payload_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    stored_owner_id = normalize_owner_key(payload.get("owner_id"))
    allowed_owner_ids = {
        normalize_owner_key(candidate)
        for candidate in (accepted_owner_ids or set())
    }
    if owner_id is not None:
        allowed_owner_ids.add(normalize_owner_key(owner_id))
    if enforce_owner and stored_owner_id not in allowed_owner_ids:
        raise ExportNotFoundError("Export payload not found.")
    return payload

def report_dir_for_token(export_token, reports_folder):
    validate_export_token(export_token)
    report_dir = Path(reports_folder) / export_token
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir

def latex_escape(value):
    '''Escape text for safe use in a LaTeX document.'''
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = "" if value is None else str(value)
    return "".join(replacements.get(character, character) for character in text)

def format_number(value, digits=3):
    metric = clean_metric(value)
    if metric is None:
        return "n/a"

    return f"{metric:.{digits}f}"

def latex_summary(summary):
    summary_text = render_analysis_summary_text(summary)
    lines = [line.strip() for line in summary_text.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return latex_escape(
            "No LLM summary was generated for this analysis. "
            "This is an associational regression analysis, not causal proof."
        )

    return "\n\n".join(latex_escape(line) for line in lines)

def write_report_graphs(payload, report_dir):
    coefficient_image = report_dir / "coefficient_stability.png"
    bootstrap_image = report_dir / "bootstrap_histogram.png"

    model_names = [point["model_name"] for point in payload["coefficient_chart"]]
    coefficients = [point["coefficient"] for point in payload["coefficient_chart"]]
    main_variable = payload["main_independent_variable"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(model_names, coefficients, marker="o", color="#111111", linewidth=2)
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_title("Coefficient Stability")
    ax.set_ylabel(f"{main_variable} coefficient")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(coefficient_image, dpi=180)
    plt.close(fig)

    bootstrap = payload["bootstrap_results"]
    samples = bootstrap["samples"]
    ci_lower, ci_upper = bootstrap["ci_95"]
    mean = bootstrap["mean"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(samples, bins=28, color="#333333", edgecolor="#ffffff")
    ax.axvline(mean, color="#111111", linewidth=2, label="Mean")
    ax.axvline(ci_lower, color="#777777", linewidth=1.5, linestyle="--", label="95% CI")
    ax.axvline(ci_upper, color="#777777", linewidth=1.5, linestyle="--")
    ax.set_title("Bootstrap Distribution")
    ax.set_xlabel(f"Bootstrapped {main_variable} coefficient")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(bootstrap_image, dpi=180)
    plt.close(fig)

    return coefficient_image, bootstrap_image

def build_latex_document(payload):
    controls = payload.get("controls") or []
    controls_text = ", ".join(controls) if controls else "None"
    bootstrap = payload["bootstrap_results"]
    ci_lower, ci_upper = bootstrap["ci_95"]

    model_rows = []
    for model in payload["models"]:
        ci_lower, ci_upper = model.get("ci_95") or [None, None]
        ci_text = f"{format_number(ci_lower, 3)} to {format_number(ci_upper, 3)}"

        model_rows.append(
            " & ".join([
                latex_escape(model.get("model_name")),
                latex_escape(model.get("formula")),
                format_number(model.get("coefficient"), 4),
                format_number(model.get("standard_error"), 4),
                format_number(model.get("t_value"), 3),
                format_number(model.get("p_value"), 4),
                latex_escape(ci_text),
                format_number(model.get("r_squared"), 4),
                format_number(model.get("adjusted_r_squared"), 4),
                format_number(model.get("rmse"), 3),
                format_number(model.get("f_statistic"), 3),
                format_number(model.get("f_p_value"), 4),
                str(model.get("n_observations", "n/a")),
            ]) + r" \\"
        )

    model_table = "\n".join(model_rows)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage[T1]{{fontenc}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.7em}}

\begin{{document}}

\begin{{center}}
{{\Large RegressAI Report}}\\
\vspace{{0.25em}}
{{\small Generated {latex_escape(payload["created_at"])}}}
\end{{center}}

\section*{{Research Question}}
{latex_escape(payload["research_question"])}

\section*{{Model Setup}}
\textbf{{Dependent variable:}} {latex_escape(payload["dependent_variable"])}\\
\textbf{{Main independent variable:}} {latex_escape(payload["main_independent_variable"])}\\
\textbf{{Controls:}} {latex_escape(controls_text)}

\textbf{{Compute mode:}} {latex_escape(payload.get("compute_mode", "CPU"))}\\
\textbf{{Compute runtime:}} {format_number(payload.get("runtime_seconds"), 3)} seconds

\section*{{Main Results}}
\textbf{{Baseline coefficient:}} {format_number(payload["baseline_coefficient"])}\\
\textbf{{Final coefficient:}} {format_number(payload["final_coefficient"])}\\
\textbf{{Coefficient change:}} {format_number(payload["coefficient_change"])}

\section*{{Coefficient Stability}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{coefficient_stability.png}}
\end{{figure}}

\section*{{Bootstrap Uncertainty}}
\textbf{{Iterations:}} {payload["bootstrap_iterations"]}\\
\textbf{{Mean coefficient:}} {format_number(bootstrap["mean"])}\\
\textbf{{Standard error:}} {format_number(bootstrap["standard_error"])}\\
\textbf{{95\% interval:}} {format_number(ci_lower)} to {format_number(ci_upper)}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{bootstrap_histogram.png}}
\end{{figure}}

\section*{{Model Progression}}
\scriptsize
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrlrrrrrr}}
\hline
Model & Formula & Coef. & SE & T & P & 95\% CI & R-sq & Adj. R-sq & RMSE & F & F P & N \\
\hline
{model_table}
\hline
\end{{tabular}}
}}
\normalsize

\section*{{LLM Research Summary}}
{latex_summary(payload["llm_summary"])}

\vfill
\textit{{This is an associational regression analysis, not causal proof.}}

\end{{document}}
"""

def ensure_report_artifacts(
    export_token,
    reports_folder,
    owner_id=None,
    accepted_owner_ids=None,
    enforce_owner=False,
):
    payload = load_export_payload(
        export_token,
        reports_folder,
        owner_id=owner_id,
        accepted_owner_ids=accepted_owner_ids,
        enforce_owner=enforce_owner,
    )
    report_dir = report_dir_for_token(export_token, reports_folder)

    try:
        write_report_graphs(payload, report_dir)
    except RuntimeError as error:
        raise ReportGenerationError(
            f"Unable to generate report graphs: {error}"
        ) from error

    tex_path = report_dir / "regression_report.tex"
    tex_path.write_text(build_latex_document(payload), encoding="utf-8")
    return report_dir, tex_path

def build_latex_zip(report_dir, tex_path):
    zip_path = report_dir / "regression_report_latex.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(tex_path, arcname=tex_path.name)
        archive.write(
            report_dir / "coefficient_stability.png",
            arcname="coefficient_stability.png"
        )
        archive.write(
            report_dir / "bootstrap_histogram.png",
            arcname="bootstrap_histogram.png"
        )
    return zip_path

def compile_pdf_report(report_dir, tex_path):
    try:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=report_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as error:
        raise ReportGenerationError(
            "pdflatex was not found. Install a LaTeX distribution such as "
            "MiKTeX or TeX Live to enable PDF export."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise ReportGenerationError(
            "pdflatex timed out while compiling the report."
        ) from error

    if result.returncode != 0:
        log_tail = (result.stdout + result.stderr)[-1600:]
        raise ReportGenerationError(f"pdflatex failed:\n{log_tail}")

    pdf_path = report_dir / "regression_report.pdf"
    if not pdf_path.exists():
        raise ReportGenerationError(
            "pdflatex finished but did not create a PDF."
        )

    return pdf_path


def cleanup_expired_exports(
    reports_folder,
    max_age_hours,
    now=None,
):
    """Delete expired export payloads and their generated report folders."""
    if max_age_hours <= 0:
        return 0

    reports_root = Path(reports_folder).resolve()
    payload_dir = (reports_root / "export_payloads").resolve()
    if not payload_dir.exists() or payload_dir.parent != reports_root:
        return 0

    cutoff = (now or datetime.now(timezone.utc)) - timedelta(
        hours=max_age_hours
    )
    removed = 0
    for payload_path in payload_dir.glob("*.json"):
        export_token = payload_path.stem
        if not re.fullmatch(r"[0-9a-f]{32}", export_token):
            continue
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(payload["created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            created_at = datetime.fromtimestamp(
                payload_path.stat().st_mtime,
                tz=timezone.utc,
            )

        if created_at >= cutoff:
            continue

        report_dir = (reports_root / export_token).resolve()
        if report_dir.parent == reports_root and report_dir.is_dir():
            shutil.rmtree(report_dir)
        payload_path.unlink(missing_ok=True)
        removed += 1

    return removed
