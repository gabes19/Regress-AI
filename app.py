import os
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, render_template, request, send_file

load_dotenv()

from config import Config
from werkzeug.utils import secure_filename
from regressionlab.services.charts_and_plots import (
    create_bootstrap_histogram_plot,
    create_coefficient_chart,
    create_coefficient_plot,
)
from regressionlab.services.regression import(
    clean_metric,
    fit_models
)
from regressionlab.services.data_processing import (
    parse_columns,
    prepare_analysis_data,
)
from regressionlab.services.bootstrap_cpu import bootstrap_coefficient
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

@app.route("/upload", methods=["POST"])
def upload():
    '''Handles user CSV upload'''
    uploaded_file = request.files.get("csv_file")

    if uploaded_file is None or uploaded_file.filename == "":
        return "No file uploaded", 400
    
    if not uploaded_file.filename.endswith(".csv"):
        return "Please upload a CSV file", 400
    
    filename = secure_filename(uploaded_file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    uploaded_file.save(save_path)
    columns = parse_columns(save_path)

    return render_template("configure.html", filename=filename, columns=columns)

@app.route("/sample/wage-education")
def sample_wage_education():
    '''Load the bundled wage/education sample dataset.'''
    sample_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        SAMPLE_DATASET_FILENAME
    )

    if not os.path.exists(sample_path):
        abort(404, description="Sample dataset not found")

    columns = parse_columns(sample_path)
    return render_template(
        "configure.html",
        filename=SAMPLE_DATASET_FILENAME,
        columns=columns
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    filename = request.form.get("filename")
    research_question = request.form.get("research_question")
    dependent_variable = request.form.get("dependent_variable")
    main_independent_variable = request.form.get("main_independent_variable")
    controls = request.form.getlist("controls")
    bootstrap_iterations = request.form.get("bootstrap_iterations")

    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(csv_path)

    def render_configuration_error(message):
        return (
            render_template(
                "configure.html",
                filename=filename,
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

    analysis_started_at = perf_counter()
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

        model_results = fit_models(
            data=prepared_data,
            dependent_variable=dependent_variable,
            main_independent_variable=main_independent_variable,
            controls=controls,
        )

        bootstrap_results = bootstrap_coefficient(
            data=prepared_data,
            main_independent_variable=main_independent_variable,
            iterations=bootstrap_iterations,
        )
    except ValueError as error:
        return render_configuration_error(str(error))

    analysis_runtime_seconds = perf_counter() - analysis_started_at

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
        compute_mode="CPU",
        runtime_seconds=analysis_runtime_seconds,
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
        llm_summary=llm_summary_data
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
        export_token=export_token
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
