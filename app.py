import os
import re
from threading import Lock
from time import monotonic
import uuid

import click
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFError, CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()

from config import Config
from regressionlab.services.charts_and_plots import (
    create_bootstrap_histogram_plot,
    create_coefficient_chart,
    create_coefficient_plot,
)
from regressionlab.services.regression import clean_metric, format_p_value
from regressionlab.services.data_processing import (
    CSVValidationError,
    parse_columns,
    prepare_analysis_data,
    validate_csv_shape,
)
from regressionlab.services.auth_service import (
    configure_google_oauth,
    current_user,
)
from regressionlab.services.compute_router import (
    ComputeUnavailableError,
    run_analysis_compute,
)
from regressionlab.services.gpu_client import RunPodClient
from regressionlab.services.gpu_usage import (
    initialize_gpu_database,
    remaining_gpu_quota,
    upsert_google_user,
)
from regressionlab.services.dataset_service import (
    DatasetError,
    DatasetNotFoundError,
    cleanup_expired_datasets,
    delete_dataset,
    load_dataset,
    store_existing_dataset,
    store_uploaded_dataset,
)
from regressionlab.services.llm_summary import (
    LLMSummaryError,
    build_fallback_summary,
    build_summary_facts,
    create_openai_client,
    generate_llm_summary,
)
from regressionlab.services.export_report import (
    ExportNotFoundError,
    ReportGenerationError,
    build_export_payload,
    store_export_payload,
    ensure_report_artifacts,
    build_latex_zip,
    compile_pdf_report,
    cleanup_expired_exports,
)


app = Flask(__name__)
app.config.from_object(Config)


def validate_production_configuration(flask_app):
    """Fail closed instead of starting with insecure production defaults."""
    if not flask_app.config.get("IS_PRODUCTION"):
        return

    problems = []
    if not os.getenv("FLASK_SECRET_KEY"):
        problems.append("FLASK_SECRET_KEY must be set")
    if not os.getenv("DATA_ROOT"):
        problems.append("DATA_ROOT must point to persistent storage")
    if not flask_app.config.get("SESSION_COOKIE_SECURE"):
        problems.append("SESSION_COOKIE_SECURE must be true")
    if not flask_app.config.get("PROXY_FIX_ENABLED"):
        problems.append("PROXY_FIX_ENABLED must be true behind the host proxy")
    if not flask_app.config.get("TRUSTED_HOSTS"):
        problems.append("TRUSTED_HOSTS must list the public hostname")
    if problems:
        raise RuntimeError(
            "Unsafe production configuration: " + "; ".join(problems)
        )


validate_production_configuration(app)
if app.config["PROXY_FIX_ENABLED"]:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

for data_folder in (
    app.config["UPLOAD_FOLDER"],
    app.config["REPORTS_FOLDER"],
    app.config["INSTANCE_FOLDER"],
):
    data_folder.mkdir(parents=True, exist_ok=True)

app.logger.setLevel(app.config["LOG_LEVEL"])
csrf = CSRFProtect(app)


def request_identity():
    user = current_user()
    return f"user:{user['id']}" if user else get_remote_address()


limiter = Limiter(
    request_identity,
    app=app,
    storage_uri=app.config["RATELIMIT_STORAGE_URI"],
    default_limits=[],
)
app.extensions["openai_client"] = create_openai_client(
    api_key=app.config.get("OPENAI_API_KEY"),
    timeout_seconds=app.config["OPENAI_TIMEOUT_SECONDS"],
)
initialize_gpu_database(app.config["GPU_USAGE_DATABASE"])
app.extensions["runpod_client"] = RunPodClient(
    endpoint_id=app.config.get("RUNPOD_ENDPOINT_ID"),
    api_key=app.config.get("RUNPOD_API_KEY"),
    wait_milliseconds=app.config["RUNPOD_WAIT_MILLISECONDS"],
    timeout_seconds=app.config["RUNPOD_HTTP_TIMEOUT_SECONDS"],
    execution_timeout_seconds=app.config["RUNPOD_EXECUTION_TIMEOUT_SECONDS"],
    price_per_second=app.config["GPU_PRICE_PER_SECOND_USD"],
    logger=app.logger,
)
app.extensions["google_oauth"] = configure_google_oauth(app)
app.extensions["artifact_cleanup_lock"] = Lock()
app.extensions["last_artifact_cleanup"] = 0.0


GUEST_OWNER_SESSION_KEY = "guest_artifact_owner"
GUEST_OWNER_PATTERN = re.compile(r"[0-9a-f]{32}")


def guest_owner_key(create=False):
    """Return this browser session's opaque guest artifact owner key."""
    guest_id = session.get(GUEST_OWNER_SESSION_KEY)
    if guest_id is not None and not GUEST_OWNER_PATTERN.fullmatch(str(guest_id)):
        session.pop(GUEST_OWNER_SESSION_KEY, None)
        guest_id = None
    if guest_id is None and create:
        guest_id = uuid.uuid4().hex
        session[GUEST_OWNER_SESSION_KEY] = guest_id
    return f"guest:{guest_id}" if guest_id else None


def artifact_access(create_guest=False):
    """Return the owner for new artifacts and all owners this session may use."""
    user = current_user()
    guest_key = guest_owner_key(create=create_guest and user is None)
    user_key = f"user:{user['id']}" if user else None
    accepted_owner_ids = {
        owner_key for owner_key in (user_key, guest_key) if owner_key
    }
    return user_key or guest_key, accepted_owner_ids


def render_upload_error(message, status_code=400):
    """Keep recoverable intake errors inside the normal upload experience."""
    return render_template("index.html", error_message=message), status_code


def run_artifact_cleanup():
    removed_datasets = cleanup_expired_datasets(
        app.config["UPLOAD_FOLDER"],
        app.config["DATA_RETENTION_HOURS"],
    )
    removed_exports = cleanup_expired_exports(
        app.config["REPORTS_FOLDER"],
        app.config["DATA_RETENTION_HOURS"],
    )
    if removed_datasets or removed_exports:
        app.logger.info(
            "Artifact cleanup removed_datasets=%s removed_exports=%s",
            removed_datasets,
            removed_exports,
        )
    return removed_datasets, removed_exports


@app.before_request
def cleanup_stale_artifacts_periodically():
    if app.testing or request.endpoint in {"static", "healthz"}:
        return None
    interval = app.config["CLEANUP_INTERVAL_SECONDS"]
    if monotonic() - app.extensions["last_artifact_cleanup"] < interval:
        return None
    lock = app.extensions["artifact_cleanup_lock"]
    if not lock.acquire(blocking=False):
        return None
    try:
        app.extensions["last_artifact_cleanup"] = monotonic()
        run_artifact_cleanup()
    except Exception as error:
        app.logger.warning(
            "Artifact cleanup failed error_type=%s", type(error).__name__
        )
    finally:
        lock.release()
    return None


@app.after_request
def set_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.plot.ly; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'self'; "
        "frame-ancestors 'none'; form-action 'self'",
    )
    if request.is_secure:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    if request.endpoint not in {"static", "healthz"}:
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.context_processor
def inject_account_context():
    user = current_user()
    quota = None
    if user:
        quota = remaining_gpu_quota(
            app.config["GPU_USAGE_DATABASE"],
            user["id"],
            app.config["GPU_DAILY_USER_LIMIT"],
            app.config["GPU_MONTHLY_USER_LIMIT"],
        )
    return {
        "current_user": user,
        "gpu_quota": quota,
        "google_login_enabled": app.extensions.get("google_oauth") is not None,
        "max_upload_megabytes": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        "data_retention_hours": app.config["DATA_RETENTION_HOURS"],
    }


@app.template_filter("metric")
def format_metric(value, digits=3):
    metric = clean_metric(value)
    if metric is None:
        return "n/a"

    return f"{metric:.{digits}f}"


@app.template_filter("p_value")
def format_p_value_filter(value, digits=4):
    return format_p_value(value, digits)

# Bundled sample dataset filename.
SAMPLE_DATASET_FILENAME = "wage_education_sample.csv"

@app.route("/")
def start():
    return render_template("index.html")


@app.route("/healthz")
@limiter.exempt
def healthz():
    return {"status": "ok"}, 200


@app.route("/login")
@limiter.limit(lambda: app.config["LOGIN_RATE_LIMIT"], exempt_when=lambda: app.testing)
def login():
    google = app.extensions.get("google_oauth")
    if google is None:
        abort(503, description="Google login is not configured.")
    return google.authorize_redirect(url_for("google_callback", _external=True))


@app.route("/auth/google/callback")
@limiter.limit(lambda: app.config["LOGIN_RATE_LIMIT"], exempt_when=lambda: app.testing)
def google_callback():
    google = app.extensions.get("google_oauth")
    if google is None:
        abort(503, description="Google login is not configured.")
    try:
        token = google.authorize_access_token()
        claims = token.get("userinfo") or google.parse_id_token(token)
        user = upsert_google_user(app.config["GPU_USAGE_DATABASE"], claims)
    except Exception as error:
        app.logger.warning("Google login failed error_type=%s", type(error).__name__)
        abort(400, description="Google login could not be verified.")
    guest_id = session.get(GUEST_OWNER_SESSION_KEY)
    session.clear()
    if guest_id and GUEST_OWNER_PATTERN.fullmatch(str(guest_id)):
        session[GUEST_OWNER_SESSION_KEY] = guest_id
    session["user"] = user
    return redirect(url_for("start"))


@app.route("/logout", methods=["POST"])
def logout():
    guest_id = session.get(GUEST_OWNER_SESSION_KEY)
    session.clear()
    if guest_id and GUEST_OWNER_PATTERN.fullmatch(str(guest_id)):
        session[GUEST_OWNER_SESSION_KEY] = guest_id
    return redirect(url_for("start"))

@app.route("/upload", methods=["POST"])
@limiter.limit(lambda: app.config["UPLOAD_RATE_LIMIT"], exempt_when=lambda: app.testing)
def upload():
    '''Handles user CSV upload'''
    uploaded_file = request.files.get("csv_file")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_upload_error("Please choose a CSV file.")

    owner_id, _ = artifact_access(create_guest=True)
    try:
        dataset = store_uploaded_dataset(
            uploaded_file,
            upload_folder=app.config["UPLOAD_FOLDER"],
            owner_id=owner_id,
        )
        validate_csv_shape(
            dataset.storage_path,
            max_rows=app.config["MAX_CSV_ROWS"],
            max_columns=app.config["MAX_CSV_COLUMNS"],
        )
        parse_columns(dataset.storage_path)
    except (DatasetError, CSVValidationError) as error:
        if "dataset" in locals():
            delete_dataset(
                dataset.dataset_id,
                upload_folder=app.config["UPLOAD_FOLDER"],
            )
        return render_upload_error(str(error))
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        if "dataset" in locals():
            delete_dataset(
                dataset.dataset_id,
                upload_folder=app.config["UPLOAD_FOLDER"],
            )
        return render_upload_error(
            "The uploaded file could not be read as CSV data."
        )

    return redirect(
        url_for("configure_dataset", dataset_id=dataset.dataset_id)
    )


@app.route("/configure/<dataset_id>")
def configure_dataset(dataset_id):
    owner_id, accepted_owner_ids = artifact_access()
    try:
        dataset = load_dataset(
            dataset_id,
            upload_folder=app.config["UPLOAD_FOLDER"],
            owner_id=owner_id,
            accepted_owner_ids=accepted_owner_ids,
            enforce_owner=True,
        )
    except DatasetNotFoundError as error:
        abort(404, description=str(error))

    try:
        columns = parse_columns(dataset.storage_path)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return render_upload_error(
            "The stored CSV could not be read. Please upload it again."
        )

    return render_template(
        "configure.html",
        dataset_id=dataset.dataset_id,
        filename=dataset.original_filename,
        columns=columns,
    )

@app.route("/sample/wage-education", methods=["POST"])
@limiter.limit(lambda: app.config["UPLOAD_RATE_LIMIT"], exempt_when=lambda: app.testing)
def sample_wage_education():
    '''Load the bundled wage/education sample dataset.'''
    sample_path = os.path.join(
        app.config["SAMPLE_DATA_FOLDER"],
        SAMPLE_DATASET_FILENAME
    )

    owner_id, _ = artifact_access(create_guest=True)
    try:
        dataset = store_existing_dataset(
            sample_path,
            upload_folder=app.config["UPLOAD_FOLDER"],
            original_filename=SAMPLE_DATASET_FILENAME,
            owner_id=owner_id,
        )
    except DatasetNotFoundError as error:
        abort(404, description=str(error))

    return redirect(
        url_for("configure_dataset", dataset_id=dataset.dataset_id)
    )


@app.route("/analyze", methods=["POST"])
@limiter.limit(lambda: app.config["ANALYSIS_RATE_LIMIT"], exempt_when=lambda: app.testing)
def analyze():
    user = current_user()
    dataset_id = request.form.get("dataset_id")
    research_question = request.form.get("research_question")
    dependent_variable = request.form.get("dependent_variable")
    main_independent_variable = request.form.get("main_independent_variable")
    controls = request.form.getlist("controls")
    bootstrap_iterations = request.form.get("bootstrap_iterations")
    gpu_opt_in_requested = bool(
        user and request.form.get("use_gpu") == "on"
    )

    owner_id, accepted_owner_ids = artifact_access()
    try:
        dataset = load_dataset(
            dataset_id,
            upload_folder=app.config["UPLOAD_FOLDER"],
            owner_id=owner_id,
            accepted_owner_ids=accepted_owner_ids,
            enforce_owner=True,
        )
    except DatasetNotFoundError as error:
        abort(404, description=str(error))

    csv_path = dataset.storage_path

    def render_configuration_error(message):
        return (
            render_template(
                "configure.html",
                dataset_id=dataset.dataset_id,
                filename=dataset.original_filename,
                columns=parse_columns(csv_path),
                error_message=message,
                research_question=research_question,
                selected_dependent_variable=dependent_variable,
                selected_main_independent_variable=(
                    main_independent_variable
                ),
                selected_controls=controls,
                selected_bootstrap_iterations=bootstrap_iterations,
                selected_gpu_opt_in=gpu_opt_in_requested,
            ),
            400,
        )

    try:
        bootstrap_iterations = int(bootstrap_iterations)
    except (TypeError, ValueError):
        return render_configuration_error(
            "Bootstrap iterations must be a whole number."
        )

    research_question = (research_question or "").strip()
    if not research_question:
        return render_configuration_error("A research question is required.")
    if len(research_question) > app.config["MAX_RESEARCH_QUESTION_LENGTH"]:
        return render_configuration_error(
            "The research question is too long. Keep it under "
            f"{app.config['MAX_RESEARCH_QUESTION_LENGTH']:,} characters."
        )
    if not (
        app.config["MIN_BOOTSTRAP_ITERATIONS"]
        <= bootstrap_iterations
        <= app.config["MAX_BOOTSTRAP_ITERATIONS"]
    ):
        return render_configuration_error(
            "Bootstrap iterations must be between "
            f"{app.config['MIN_BOOTSTRAP_ITERATIONS']:,} and "
            f"{app.config['MAX_BOOTSTRAP_ITERATIONS']:,}."
        )

    try:
        df = pd.read_csv(csv_path)
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return render_configuration_error(
            "The stored dataset could not be read. Upload it again."
        )

    try:
        prepared_data = prepare_analysis_data(
            df=df,
            dependent_variable=dependent_variable,
            main_independent_variable=main_independent_variable,
            controls=controls,
        )

        compute_result = run_analysis_compute(
            data=prepared_data,
            dependent_variable=dependent_variable,
            main_independent_variable=main_independent_variable,
            controls=controls,
            bootstrap_iterations=bootstrap_iterations,
            config=app.config,
            gpu_client=app.extensions.get("runpod_client"),
            user_id=(user or {}).get("id"),
            logger=app.logger,
            gpu_opt_in=gpu_opt_in_requested,
        )
        model_results = compute_result.model_results
        bootstrap_results = compute_result.bootstrap_results
    except (ValueError, ComputeUnavailableError) as error:
        return render_configuration_error(str(error))

    # From this point onward, the results and exports use derived values only.
    # Fail closed if the raw upload cannot be removed so the results page never
    # claims successful deletion when the CSV may still be stored.
    try:
        delete_dataset(
            dataset.dataset_id,
            upload_folder=app.config["UPLOAD_FOLDER"],
        )
    except OSError:
        app.logger.exception(
            "Analyzed dataset deletion failed dataset_id=%s",
            dataset.dataset_id,
        )
        abort(
            500,
            description=(
                "The analysis finished, but deletion of the uploaded CSV "
                "could not be confirmed."
            ),
        )
    app.logger.info(
        "Analyzed dataset deleted dataset_id=%s",
        dataset.dataset_id,
    )

    compute_mode = compute_result.compute_mode

    baseline_coefficient = model_results[0]["coefficient"]
    final_coefficient = model_results[-1]["coefficient"]
    coefficient_change = final_coefficient - baseline_coefficient
    coefficient_chart = create_coefficient_chart(model_results)
    coefficient_plot_html = create_coefficient_plot(
        coefficient_chart,
        main_independent_variable
    )
    bootstrap_histogram_html = create_bootstrap_histogram_plot(
        bootstrap_results,
        main_independent_variable
    )
    summary_facts = build_summary_facts(
        research_question=research_question,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
        model_results=model_results,
        bootstrap_results=bootstrap_results,
        bootstrap_iterations=bootstrap_iterations,
        compute_mode=compute_mode,
        runtime_seconds=compute_result.runtime_seconds,
    )
    if user:
        try:
            llm_summary = generate_llm_summary(
                summary_facts,
                client=app.extensions.get("openai_client"),
                model=app.config["OPENAI_MODEL"],
                pricing=app.config["OPENAI_MODEL_PRICING"],
                max_output_tokens=app.config["OPENAI_MAX_OUTPUT_TOKENS"],
                logger=app.logger,
            )
        except LLMSummaryError:
            llm_summary = build_fallback_summary(summary_facts)
    else:
        llm_summary = build_fallback_summary(summary_facts)

    llm_summary_data = llm_summary.model_dump(mode="json")
    export_payload = build_export_payload(
        research_question=research_question,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
        bootstrap_iterations=bootstrap_iterations,
        model_results=model_results,
        baseline_coefficient=baseline_coefficient,
        final_coefficient=final_coefficient,
        coefficient_change=coefficient_change,
        coefficient_chart=coefficient_chart,
        bootstrap_results=bootstrap_results,
        llm_summary=llm_summary_data,
        compute_mode=compute_mode,
        runtime_seconds=compute_result.runtime_seconds,
    )
    export_token = store_export_payload(
        export_payload,
        reports_folder=app.config["REPORTS_FOLDER"],
        owner_id=owner_id,
    )

    return render_template(
        "results.html",
        research_question=research_question,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
        bootstrap_iterations=bootstrap_iterations,
        models=model_results,
        baseline_coefficient=baseline_coefficient,
        final_coefficient=final_coefficient,
        coefficient_change=coefficient_change,
        coefficient_chart=coefficient_chart,
        coefficient_plot_html=coefficient_plot_html,
        bootstrap_results=bootstrap_results,
        bootstrap_histogram_html=bootstrap_histogram_html,
        llm_summary=llm_summary_data,
        export_token=export_token,
        compute_mode=compute_mode,
        compute_runtime_seconds=compute_result.runtime_seconds,
        gpu_name=compute_result.gpu_name,
        llm_enabled_for_user=bool(user),
    )


@app.route("/export/latex/<export_token>")
@limiter.limit(lambda: app.config["EXPORT_RATE_LIMIT"], exempt_when=lambda: app.testing)
def export_latex(export_token):
    owner_id, accepted_owner_ids = artifact_access()
    try:
        report_dir, tex_path = ensure_report_artifacts(
            export_token,
            reports_folder=app.config["REPORTS_FOLDER"],
            owner_id=owner_id,
            accepted_owner_ids=accepted_owner_ids,
            enforce_owner=True,
        )
        zip_path = build_latex_zip(report_dir, tex_path)
    except ExportNotFoundError as error:
        abort(404, description=str(error))
    except ReportGenerationError as error:
        abort(500, description=str(error))

    return send_file(
        zip_path,
        as_attachment=True,
        download_name="regressai_latex.zip",
        mimetype="application/zip",
    )

@app.route("/export/pdf/<export_token>")
@limiter.limit(lambda: app.config["EXPORT_RATE_LIMIT"], exempt_when=lambda: app.testing)
def export_pdf(export_token):
    owner_id, accepted_owner_ids = artifact_access()
    try:
        report_dir, tex_path = ensure_report_artifacts(
            export_token,
            reports_folder=app.config["REPORTS_FOLDER"],
            owner_id=owner_id,
            accepted_owner_ids=accepted_owner_ids,
            enforce_owner=True,
        )
        pdf_path = compile_pdf_report(report_dir, tex_path)
    except ExportNotFoundError as error:
        abort(404, description=str(error))
    except ReportGenerationError as error:
        abort(500, description=str(error))

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="regressai_report.pdf",
        mimetype="application/pdf",
    )


@app.cli.command("cleanup-artifacts")
def cleanup_artifacts_command():
    """Delete datasets and reports outside the retention window."""
    datasets, exports = run_artifact_cleanup()
    click.echo(f"Removed {datasets} datasets and {exports} exports.")


@app.errorhandler(CSRFError)
def handle_csrf_error(error):
    return render_template(
        "error.html",
        status_code=400,
        title="Request expired",
        message="Refresh the page and try again.",
    ), 400


@app.errorhandler(413)
def handle_upload_too_large(error):
    return render_upload_error(
        (
            f"CSV files must be no larger than "
            f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."
        ),
        413,
    )


@app.errorhandler(429)
def handle_rate_limit(error):
    return render_template(
        "error.html",
        status_code=429,
        title="Too many requests",
        message="Please wait before trying again.",
    ), 429


@app.errorhandler(404)
def handle_not_found(error):
    return render_template(
        "error.html",
        status_code=404,
        title="Not found",
        message=getattr(error, "description", "The requested page was not found."),
    ), 404


@app.errorhandler(500)
def handle_internal_error(error):
    app.logger.error("Unhandled application error error_type=%s", type(error).__name__)
    return render_template(
        "error.html",
        status_code=500,
        title="Something went wrong",
        message="The request could not be completed. Please try again.",
    ), 500
