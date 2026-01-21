"""
Autograder Agent - Streamlit Frontend

A web interface for automatically grading student submissions.
"""
import streamlit as st
import pandas as pd
import tempfile
import os
from io import BytesIO

from config import Config
from llm_client import LLMClient
from file_processor import FileProcessor, SubmissionContent
from grading_engine import GradingEngine, RubricCriteria, GradingResult

# Page configuration
st.set_page_config(
    page_title="Autograder Agent",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .score-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    /* Rubric styling */
    .rubric-item {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "grading_results" not in st.session_state:
        st.session_state.grading_results = []
    if "rubric_criteria" not in st.session_state:
        st.session_state.rubric_criteria = []
    if "rubric_items" not in st.session_state:
        # Default rubric items with unique IDs
        st.session_state.rubric_items = [
            {"id": 1, "name": "Data Loading", "points": 10, "description": "Correctly loads and inspects the dataset"},
            {"id": 2, "name": "Exploratory Data Analysis", "points": 15, "description": "Performs meaningful EDA with visualizations"},
            {"id": 3, "name": "Feature Engineering", "points": 10, "description": "Creates relevant features for the model"},
            {"id": 4, "name": "Model Implementation", "points": 20, "description": "Correctly implements the required model"},
            {"id": 5, "name": "Model Evaluation", "points": 15, "description": "Evaluates model with appropriate metrics"},
            {"id": 6, "name": "Code Quality", "points": 10, "description": "Clean, well-documented code"},
            {"id": 7, "name": "Presentation", "points": 20, "description": "Clear presentation of findings and conclusions"},
        ]
    if "rubric_id_counter" not in st.session_state:
        st.session_state.rubric_id_counter = 8
    if "problem_statement" not in st.session_state:
        st.session_state.problem_statement = ""
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = ""
    if "submissions" not in st.session_state:
        st.session_state.submissions = []
    if "provider" not in st.session_state:
        st.session_state.provider = Config.LLM_PROVIDER
    if "model" not in st.session_state:
        st.session_state.model = Config.get_model()
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = ""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = ""


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.markdown("## Settings")

        # Provider selection
        st.markdown("### LLM Provider")

        ollama_status = Config.is_ollama_available()

        provider = st.selectbox(
            "Select Provider",
            options=["ollama", "openai", "anthropic"],
            index=["ollama", "openai", "anthropic"].index(st.session_state.provider),
            key="provider_select"
        )

        if provider == "ollama":
            if ollama_status:
                st.success("Ollama is running")
            else:
                st.error("Ollama not detected")

        # Model selection
        st.markdown("### Model")
        available_models = LLMClient.get_available_models(provider)

        if available_models:
            model = st.selectbox(
                "Select Model",
                options=available_models,
                index=0,
                key="model_select"
            )
        else:
            model = st.text_input(
                "Model Name",
                value=st.session_state.model,
                key="model_input"
            )

        # API Keys (stored in session state only, not persisted)
        if provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.openai_api_key,
                key="openai_key"
            )
            st.session_state.openai_api_key = api_key

        elif provider == "anthropic":
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.anthropic_api_key,
                key="anthropic_key"
            )
            st.session_state.anthropic_api_key = api_key

        if st.button("Apply Settings", type="primary"):
            st.session_state.provider = provider
            st.session_state.model = model
            st.success("Settings applied!")
            st.rerun()

        # Test connection
        if st.button("Test Connection"):
            with st.spinner("Testing..."):
                # Pass API key from session state
                api_key = None
                if provider == "openai":
                    api_key = st.session_state.openai_api_key
                elif provider == "anthropic":
                    api_key = st.session_state.anthropic_api_key
                result = LLMClient.test_connection(provider, model, api_key=api_key)
                if result["success"]:
                    st.success("Connected!")
                else:
                    st.error(f"Failed: {result['error']}")

        st.divider()

        # Clear results
        if st.button("Clear All Results", use_container_width=True):
            st.session_state.grading_results = []
            st.session_state.submissions = []
            st.rerun()


def render_input_section():
    """Render the input section for problem statement, dataset, and rubric."""
    st.markdown("## Setup Grading")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Problem Statement")

        problem_input_type = st.radio(
            "Input method",
            ["Text Box", "Upload File"],
            key="problem_input_type",
            horizontal=True
        )

        if problem_input_type == "Text Box":
            problem_statement = st.text_area(
                "Enter the problem statement/requirements",
                value=st.session_state.problem_statement,
                height=200,
                placeholder="Describe what students were asked to do..."
            )
        else:
            problem_file = st.file_uploader(
                "Upload problem statement",
                type=["txt", "md", "pdf"],
                key="problem_file"
            )
            if problem_file:
                if problem_file.name.endswith('.pdf'):
                    st.warning("PDF support requires PyPDF2. Using filename as placeholder.")
                    problem_statement = f"[Problem from: {problem_file.name}]"
                else:
                    problem_statement = problem_file.read().decode('utf-8')
            else:
                problem_statement = st.session_state.problem_statement

        st.session_state.problem_statement = problem_statement

        # Dataset upload
        st.markdown("### Dataset (Optional)")
        dataset_file = st.file_uploader(
            "Upload the dataset used in the assignment",
            type=["csv", "xlsx", "xls", "json"],
            key="dataset_file"
        )

        if dataset_file:
            processor = FileProcessor()
            dataset_info = processor.process_uploaded_dataset(dataset_file)
            if dataset_info.get("success"):
                st.session_state.dataset_info = dataset_info.get("description", "")

                # Save dataset to a temp file for code execution
                temp_dir = tempfile.gettempdir()
                dataset_path = os.path.join(temp_dir, dataset_file.name)
                dataset_file.seek(0)  # Reset file pointer
                with open(dataset_path, 'wb') as f:
                    f.write(dataset_file.read())
                st.session_state.dataset_path = dataset_path

                with st.expander("Dataset Preview"):
                    st.text(dataset_info.get("description", ""))
            else:
                st.error(dataset_info.get("error", "Failed to load dataset"))

    with col2:
        render_rubric_section()


def delete_rubric_item(item_id):
    """Delete a rubric item by its unique ID."""
    st.session_state.rubric_items = [
        item for item in st.session_state.rubric_items if item["id"] != item_id
    ]


def render_rubric_section():
    """Render the interactive rubric management section."""
    st.markdown("### Grading Rubric")

    # Calculate total points
    total_points = sum(item["points"] for item in st.session_state.rubric_items)
    st.info(f"Total Points: **{total_points}** | Criteria: **{len(st.session_state.rubric_items)}**")

    # Column headers
    if st.session_state.rubric_items:
        header_col1, header_col2, header_col3, header_col4 = st.columns([3, 1, 4, 0.5])
        with header_col1:
            st.markdown("**Criterion Name**")
        with header_col2:
            st.markdown("**Points**")
        with header_col3:
            st.markdown("**Description**")
        with header_col4:
            st.markdown("")

    # Render each rubric criterion
    for idx, item in enumerate(st.session_state.rubric_items):
        item_id = item["id"]
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 4, 0.5])

            with col1:
                new_name = st.text_input(
                    "Criterion",
                    value=item["name"],
                    key=f"rubric_name_{item_id}",
                    label_visibility="collapsed",
                    placeholder="Criterion name"
                )
                st.session_state.rubric_items[idx]["name"] = new_name

            with col2:
                new_points = st.number_input(
                    "Points",
                    value=item["points"],
                    min_value=1,
                    max_value=100,
                    key=f"rubric_points_{item_id}",
                    label_visibility="collapsed"
                )
                st.session_state.rubric_items[idx]["points"] = new_points

            with col3:
                new_desc = st.text_input(
                    "Description",
                    value=item["description"],
                    key=f"rubric_desc_{item_id}",
                    label_visibility="collapsed",
                    placeholder="What to look for..."
                )
                st.session_state.rubric_items[idx]["description"] = new_desc

            with col4:
                st.button("X", key=f"delete_rubric_{item_id}", help="Delete this criterion",
                          on_click=delete_rubric_item, args=(item_id,))

    # Add new criterion button
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("+ Add Criterion", use_container_width=True):
            new_id = st.session_state.rubric_id_counter
            st.session_state.rubric_id_counter += 1
            st.session_state.rubric_items.append({
                "id": new_id,
                "name": f"New Criterion {len(st.session_state.rubric_items) + 1}",
                "points": 10,
                "description": "Description of what to evaluate"
            })
            st.rerun()

    with col2:
        if st.button("Reset to Default", use_container_width=True):
            st.session_state.rubric_items = [
                {"id": 1, "name": "Data Loading", "points": 10, "description": "Correctly loads and inspects the dataset"},
                {"id": 2, "name": "Exploratory Data Analysis", "points": 15, "description": "Performs meaningful EDA with visualizations"},
                {"id": 3, "name": "Feature Engineering", "points": 10, "description": "Creates relevant features for the model"},
                {"id": 4, "name": "Model Implementation", "points": 20, "description": "Correctly implements the required model"},
                {"id": 5, "name": "Model Evaluation", "points": 15, "description": "Evaluates model with appropriate metrics"},
                {"id": 6, "name": "Code Quality", "points": 10, "description": "Clean, well-documented code"},
                {"id": 7, "name": "Presentation", "points": 20, "description": "Clear presentation of findings and conclusions"},
            ]
            st.session_state.rubric_id_counter = 8
            st.rerun()

    with col3:
        if st.button("Clear All", use_container_width=True):
            st.session_state.rubric_items = []
            st.session_state.rubric_id_counter = 1
            st.rerun()

    # Convert to RubricCriteria objects for grading
    if st.session_state.rubric_items:
        st.session_state.rubric_criteria = [
            RubricCriteria(
                name=item["name"],
                description=item["description"],
                max_points=item["points"]
            )
            for item in st.session_state.rubric_items
            if item["name"].strip()  # Only include items with names
        ]

        # Show preview
        with st.expander("Rubric Preview"):
            for item in st.session_state.rubric_items:
                st.markdown(f"**{item['name']}** ({item['points']} pts): {item['description']}")
    else:
        st.session_state.rubric_criteria = []
        st.warning("Add at least one criterion to the rubric")


def render_submission_section():
    """Render the submission upload section."""
    st.markdown("## Upload Submissions")

    uploaded_files = st.file_uploader(
        "Upload student submissions",
        type=["py", "ipynb", "zip", "pptx"],
        accept_multiple_files=True,
        key="submissions_upload"
    )

    if uploaded_files:
        processor = FileProcessor()
        submissions = []

        with st.expander(f"{len(uploaded_files)} files uploaded"):
            for file in uploaded_files:
                content = processor.process_uploaded_file(file)
                submissions.append(content)

                if content.errors:
                    st.error(f"{file.name}: {content.errors[0]}")
                else:
                    st.success(f"{file.name} ({content.file_type})")

                    # Show preview
                    if content.code:
                        with st.expander(f"Code preview - {file.name}"):
                            st.code(content.code[:1000] + "..." if len(content.code) > 1000 else content.code)

        st.session_state.submissions = submissions

    return len(st.session_state.submissions) > 0


def render_grading_section():
    """Render the grading section."""
    st.markdown("## Grade Submissions")

    # Validation
    can_grade = True
    issues = []

    if not st.session_state.problem_statement:
        issues.append("Problem statement is required")
        can_grade = False

    if not st.session_state.rubric_criteria:
        issues.append("Rubric is required")
        can_grade = False

    if not st.session_state.submissions:
        issues.append("No submissions uploaded")
        can_grade = False

    for issue in issues:
        st.warning(issue)

    if can_grade:
        st.success(f"Ready to grade {len(st.session_state.submissions)} submissions")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("Start Grading", type="primary", use_container_width=True):
                grade_submissions()

        with col2:
            if st.button("Grade Sample (First Only)", use_container_width=True):
                grade_submissions(sample_only=True)


def grade_submissions(sample_only=False):
    """Run the grading process."""
    submissions = st.session_state.submissions
    if sample_only:
        submissions = submissions[:1]

    # Get the appropriate API key based on provider
    api_key = None
    if st.session_state.provider == "openai":
        api_key = st.session_state.openai_api_key
    elif st.session_state.provider == "anthropic":
        api_key = st.session_state.anthropic_api_key

    engine = GradingEngine(
        provider=st.session_state.provider,
        model=st.session_state.model,
        api_key=api_key
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Tool-assisted grading (ReAct) enabled...")

    results = []

    for i, submission in enumerate(submissions):
        progress = (i + 1) / len(submissions)
        progress_bar.progress(progress)
        status_text.text(f"Grading: {submission.filename} ({i + 1}/{len(submissions)})")

        try:
            result = engine.grade_submission(
                submission=submission,
                problem_statement=st.session_state.problem_statement,
                dataset_info=st.session_state.dataset_info,
                rubric=st.session_state.rubric_criteria,
                dataset_path=st.session_state.dataset_path
            )
            results.append(result)
        except Exception as e:
            results.append(GradingResult(
                filename=submission.filename,
                errors=[str(e)]
            ))

    progress_bar.progress(1.0)
    status_text.text("Grading complete!")

    st.session_state.grading_results = results
    st.rerun()


def render_results_section():
    """Render the results section with dual reports (concise and detailed)."""
    if not st.session_state.grading_results:
        return

    st.markdown("## Grading Results")

    results = st.session_state.grading_results

    # Create DataFrame
    engine = GradingEngine()
    df = engine.results_to_dataframe(results)

    if df.empty:
        st.warning("No results to display")
        return

    # Report type selector
    report_type = st.radio(
        "Report Type",
        ["Concise Report", "Detailed Report"],
        horizontal=True,
        key="report_type"
    )

    st.divider()

    if report_type == "Concise Report":
        render_concise_report(results, df)
    else:
        render_detailed_report(results, df)


def render_concise_report(results, df):
    """Render a concise summary report."""
    st.markdown("### Summary")

    # Score table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export options
    st.markdown("### Export")

    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "grading_results.csv",
            "text/csv",
            use_container_width=True
        )

    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Grades')
        buffer.seek(0)

        st.download_button(
            "Download Excel",
            buffer,
            "grading_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # Brief feedback for each submission
    st.markdown("### Quick Feedback")

    for result in results:
        if result.errors:
            st.error(f"**{result.filename}**: Error - {result.errors[0]}")
        else:
            status = "PASS" if result.percentage >= 60 else "NEEDS IMPROVEMENT"
            st.markdown(f"**{result.filename}**: {result.total_score}/{result.max_total} ({result.percentage:.1f}%) - {status}")
            if result.overall_feedback:
                st.caption(result.overall_feedback[:200] + "..." if len(result.overall_feedback) > 200 else result.overall_feedback)


def render_detailed_report(results, df):
    """Render a detailed report with full feedback and evidence."""
    st.markdown("### Detailed Analysis")

    # Export options first
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "grading_results.csv",
            "text/csv",
            use_container_width=True
        )

    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Grades')
        buffer.seek(0)

        st.download_button(
            "Download Excel",
            buffer,
            "grading_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.divider()

    # Debug mode toggle
    show_debug = st.checkbox("Show debug info (raw LLM responses, orchestration traces)", value=False)

    # Detailed feedback per submission
    for result in results:
        with st.expander(f"{result.filename} - {result.percentage:.1f}%", expanded=True):
            if result.errors:
                for error in result.errors:
                    st.error(error)

            # Check if all scores are zero (parsing issue)
            all_zero = result.scores and all(s.score == 0 for s in result.scores.values())
            if all_zero:
                st.warning("All scores are 0. This may indicate a parsing issue. Enable debug mode below to see the raw LLM response.")

            # Score breakdown by criterion
            st.markdown("#### Score Breakdown")

            if result.scores:
                for name, score in result.scores.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        pct = score.percentage
                        if pct >= 80:
                            indicator = "[HIGH]"
                        elif pct >= 60:
                            indicator = "[MED]"
                        else:
                            indicator = "[LOW]"
                        st.markdown(f"{indicator} **{name}**: {score.score}/{score.max_points}")
                    with col2:
                        st.markdown(f"_{score.feedback}_")

                st.divider()
                st.markdown(f"**Total: {result.total_score}/{result.max_total} ({result.percentage:.1f}%)**")

            # Overall feedback
            if result.overall_feedback:
                st.markdown("#### Overall Feedback")
                st.info(result.overall_feedback)

            # Tool evidence
            if hasattr(result, 'used_tools') and result.used_tools:
                st.markdown("#### Grading Method")
                st.caption("Tool-Assisted (ReAct Pattern)")

                if hasattr(result, 'tool_evidence') and result.tool_evidence:
                    st.markdown("#### Tool Evidence")
                    st.text(result.tool_evidence)

            # Debug info
            if show_debug:
                if hasattr(result, 'raw_response') and result.raw_response:
                    st.markdown("#### Raw LLM Response")
                    st.code(result.raw_response, language="json")

                if hasattr(result, 'orchestration_trace') and result.orchestration_trace:
                    st.markdown("#### Orchestration Trace (ReAct Steps)")
                    st.text(result.orchestration_trace)


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">Autograder Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666;">AI-powered automatic grading for student submissions</p>',
        unsafe_allow_html=True
    )

    # Render sidebar
    render_sidebar()

    # Main content
    render_input_section()

    st.divider()

    has_submissions = render_submission_section()

    if has_submissions:
        st.divider()
        render_grading_section()

    if st.session_state.grading_results:
        st.divider()
        render_results_section()


if __name__ == "__main__":
    main()
