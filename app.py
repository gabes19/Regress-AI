import os
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

load_dotenv()

from config import Config
from regressionlab.services.charts_and_plots import (
    create_bootstrap_histogram_plot,
    create_coefficient_chart,
    create_coefficient_plot,
)
from regressionlab.services.regression import clean_metric
from regressionlab.services.data_processing import (
    parse_columns,
    prepare_analysis_data,
)
from regressionlab.services.auth_service import (
    configure_google_oauth,
    current_user,
    logout_csrf_token,
    validate_logout_csrf,
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
)


app = Flask(__name__)
app.config.from_object(Config)
app.logger.setLevel(app.config["LOG_LEVEL"])
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
        "logout_csrf_token": logout_csrf_token(),
    }


@app.template_filter("metric")
def format_metric(value, digits=3):
    metric = clean_metric(value)
    if metric is None:
        return "n/a"

    return f"{metric:.{digits}f}"

# Bundled sample dataset filename.
SAMPLE_DATASET_FILENAME = "wage_education_sample.csv"

@app.route("/")
def start():
    return render_template("index.html")


@app.route("/login")
def login():
    google = app.extensions.get("google_oauth")
    if google is None:
        abort(503, description="Google login is not configured.")
    return google.authorize_redirect(url_for("google_callback", _external=True))


@app.route("/auth/google/callback")
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
    session.clear()
    session["user"] = user
    return redirect(url_for("start"))


@app.route("/logout", methods=["POST"])
def logout():
    if not validate_logout_csrf(request.form.get("csrf_token")):
        abort(400, description="Invalid logout request.")
    session.clear()
    return redirect(url_for("start"))

@app.route("/upload", methods=["POST"])
def upload():
    '''Handles user CSV upload'''
    uploaded_file = request.files.get("csv_file")

    if uploaded_file is None or uploaded_file.filename == "":
        return "No file uploaded", 400

    try:
        dataset = store_uploaded_dataset(
            uploaded_file,
            upload_folder=app.config["UPLOAD_FOLDER"],
        )
        parse_columns(dataset.storage_path)
    except DatasetError as error:
        return str(error), 400
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        if "dataset" in locals():
            delete_dataset(
                dataset.dataset_id,
                upload_folder=app.config["UPLOAD_FOLDER"],
            )
        return "The uploaded file could not be read as CSV data.", 400

    return redirect(
        url_for("configure_dataset", dataset_id=dataset.dataset_id)
    )


@app.route("/configure/<dataset_id>")
def configure_dataset(dataset_id):
    try:
        dataset = load_dataset(
            dataset_id,
            upload_folder=app.config["UPLOAD_FOLDER"],
        )
    except DatasetNotFoundError as error:
        abort(404, description=str(error))

    return render_template(
        "configure.html",
        dataset_id=dataset.dataset_id,
        filename=dataset.original_filename,
        columns=parse_columns(dataset.storage_path),
    )

@app.route("/sample/wage-education")
def sample_wage_education():
    '''Load the bundled wage/education sample dataset.'''
    sample_path = os.path.join(
        app.config["SAMPLE_DATA_FOLDER"],
        SAMPLE_DATASET_FILENAME
    )

    try:
        dataset = store_existing_dataset(
            sample_path,
            upload_folder=app.config["UPLOAD_FOLDER"],
            original_filename=SAMPLE_DATASET_FILENAME,
        )
    except DatasetNotFoundError as error:
        abort(404, description=str(error))

    return redirect(
        url_for("configure_dataset", dataset_id=dataset.dataset_id)
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    dataset_id = request.form.get("dataset_id")
    research_question = request.form.get("research_question")
    dependent_variable = request.form.get("dependent_variable")
    main_independent_variable = request.form.get("main_independent_variable")
    controls = request.form.getlist("controls")
    bootstrap_iterations = request.form.get("bootstrap_iterations")

    try:
        dataset = load_dataset(
            dataset_id,
            upload_folder=app.config["UPLOAD_FOLDER"],
        )
    except DatasetNotFoundError as error:
        abort(404, description=str(error))

    csv_path = dataset.storage_path
    df = pd.read_csv(csv_path)

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
            ),
            400,
        )

    try:
        bootstrap_iterations = int(bootstrap_iterations)
    except (TypeError, ValueError):
        return render_configuration_error(
            "Bootstrap iterations must be a whole number."
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
            user_id=(current_user() or {}).get("id"),
            logger=app.logger,
        )
        model_results = compute_result.model_results
        bootstrap_results = compute_result.bootstrap_results
    except (ValueError, ComputeUnavailableError) as error:
        return render_configuration_error(str(error))

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
    )


@app.route("/export/latex/<export_token>")
def export_latex(export_token):
    try:
        report_dir, tex_path = ensure_report_artifacts(
            export_token,
            reports_folder=app.config["REPORTS_FOLDER"],
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
def export_pdf(export_token):
    try:
        report_dir, tex_path = ensure_report_artifacts(
            export_token,
            reports_folder=app.config["REPORTS_FOLDER"],
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
