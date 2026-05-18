import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.models.predict import load_artifacts, predict_salary

PAGE_TITLE = "Job Salary Prediction"
PAGE_ICON = "💼"

JOB_TITLES = [
    "AI Engineer",
    "Backend Developer",
    "Business Analyst",
    "Cloud Engineer",
    "Cybersecurity Analyst",
    "Data Analyst",
    "Data Scientist",
    "DevOps Engineer",
    "Frontend Developer",
    "Machine Learning Engineer",
    "Product Manager",
    "Software Engineer",
]

EDUCATION_LEVELS = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
INDUSTRIES = [
    "Consulting",
    "Education",
    "Finance",
    "Government",
    "Healthcare",
    "Manufacturing",
    "Media",
    "Retail",
    "Technology",
    "Telecom",
]
COMPANY_SIZES = ["Enterprise", "Large", "Medium", "Small", "Startup"]
LOCATIONS = [
    "Australia",
    "Canada",
    "Germany",
    "India",
    "Netherlands",
    "Singapore",
    "Sweden",
    "UK",
    "USA",
    "Remote",
]
REMOTE_OPTIONS = ["Yes", "Hybrid", "No"]
DEFAULT_INPUTS = {
    "education_level": "Bachelor",
    "industry": "Technology",
    "company_size": "Medium",
    "remote_work": "Yes",
    "skills_count": 10,
    "certifications": 2,
}


def inject_glassmorphism_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0f172a 100%);
        }
        .stApp {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.95));
        }
        .glass-card,
        .glass-sidebar {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            padding: 1.5rem;
        }
        .glass-card {
            margin-bottom: 1.75rem;
        }
        .glass-sidebar {
            margin: 0.75rem 0 1rem 0;
        }
        .salary-value {
            font-size: 3rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
            line-height: 1.05;
        }
        .metric-label {
            color: #cbd5e1;
            margin-top: 0.25rem;
            font-size: 0.95rem;
        }
        .stButton>button {
            background: rgba(255, 255, 255, 0.15) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.24) !important;
            box-shadow: none !important;
        }
        .stSelectbox>div>div>div,
        .stNumberInput>div>div>input,
        .stSlider>div>div>input {
            background: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
        }
        .streamlit-expanderHeader {
            color: #ffffff !important;
        }
        .css-1v3fvcr {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_artifacts() -> dict:
    return load_artifacts("config/config.yaml")


@st.cache_data
def load_market_data() -> pd.DataFrame:
    raw_path = "data/raw/job_salary_prediction_dataset.csv"
    return pd.read_csv(raw_path)


def format_currency(value: float) -> str:
    return f"Rp {value:,.0f}"


def build_input_panel(advanced_mode: bool) -> dict:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Job Profile Input")

    left_col, right_col = st.columns(2)
    with left_col:
        job_title = st.selectbox("Job Title", JOB_TITLES, index=JOB_TITLES.index("Data Scientist"))
        experience_years = st.slider("Experience Years", 1, 20, 10)
        education_level = (
            st.selectbox("Education Level", EDUCATION_LEVELS, index=EDUCATION_LEVELS.index("Bachelor"))
            if advanced_mode
            else DEFAULT_INPUTS["education_level"]
        )
        industry = (
            st.selectbox("Industry", INDUSTRIES, index=INDUSTRIES.index("Technology"))
            if advanced_mode
            else DEFAULT_INPUTS["industry"]
        )

    with right_col:
        location = st.selectbox("Location", LOCATIONS, index=LOCATIONS.index("USA"))
        company_size = (
            st.selectbox("Company Size", COMPANY_SIZES, index=COMPANY_SIZES.index("Medium"))
            if advanced_mode
            else DEFAULT_INPUTS["company_size"]
        )
        remote_work = (
            st.selectbox("Remote Work", REMOTE_OPTIONS, index=REMOTE_OPTIONS.index("Yes"))
            if advanced_mode
            else DEFAULT_INPUTS["remote_work"]
        )
        skills_count = (
            st.slider("Skills Count", 0, 19, DEFAULT_INPUTS["skills_count"])
            if advanced_mode
            else DEFAULT_INPUTS["skills_count"]
        )
        certifications = (
            st.slider("Certifications", 0, 5, DEFAULT_INPUTS["certifications"])
            if advanced_mode
            else DEFAULT_INPUTS["certifications"]
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "job_title": job_title,
        "education_level": education_level,
        "industry": industry,
        "company_size": company_size,
        "location": location,
        "remote_work": remote_work,
        "experience_years": experience_years,
        "skills_count": skills_count,
        "certifications": certifications,
    }


def plot_salary_distribution(salary_series: pd.Series, prediction: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(salary_series, bins=40, kde=True, color="#7c3aed", alpha=0.5, ax=ax)
    ax.axvline(prediction, color="#ff3864", linewidth=3, linestyle="--")
    ax.set_title("Market Salary Distribution", color="#f8fafc")
    ax.set_xlabel("Salary (IDR)", color="#e2e8f0")
    ax.set_ylabel("Count", color="#e2e8f0")
    ax.tick_params(colors="#e2e8f0")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_color("#cbd5e1")
    st.pyplot(fig)


def plot_feature_importance(artifacts: dict) -> None:
    config = artifacts["config"]
    model = artifacts["model"]
    feature_names = config["features"]["categorical_columns"] + config["features"]["numeric_columns"]
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        st.warning("Model tidak memiliki atribut feature_importances_.")
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=importance_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance", color="#f8fafc")
    ax.set_xlabel("Importance", color="#e2e8f0")
    ax.set_ylabel("")
    ax.tick_params(colors="#e2e8f0")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_color("#cbd5e1")
    ax.grid(axis="x", color="rgba(255,255,255,0.1)")
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    st.pyplot(fig)


def render_sidebar() -> bool:
    st.sidebar.markdown("<div class='glass-sidebar'>", unsafe_allow_html=True)
    st.sidebar.title("Control Panel")
    advanced_mode = st.sidebar.checkbox("Advanced Mode", value=False)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    return advanced_mode


def render_main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    inject_glassmorphism_css()

    st.markdown("# Job Salary Prediction")
    st.markdown("Aplikasi ini memprediksi gaji berdasarkan deskripsi pekerjaan dan profil profesional dengan tema Glassmorphism Dashboard.")

    advanced_mode = render_sidebar()
    artifacts = None

    try:
        artifacts = load_model_artifacts()
    except FileNotFoundError as error:
        st.error(f"Artifact load failed: {error}")
        return
    except Exception as error:
        st.error(f"Tidak dapat memuat model: {error}")
        return

    raw_input = build_input_panel(advanced_mode)

    if st.button("Predict Salary"):
        try:
            prediction = predict_salary(raw_input, artifacts)[0]
            base_salary = float(prediction)
        except Exception as error:
            st.error(f"Error saat inferensi: {error}")
            return

        cert_input = raw_input.copy()
        cert_input["certifications"] = min(cert_input["certifications"] + 1, 5)
        skills_input = raw_input.copy()
        skills_input["skills_count"] = min(skills_input["skills_count"] + 2, 19)

        try:
            cert_salary = predict_salary(cert_input, artifacts)[0]
            skills_salary = predict_salary(skills_input, artifacts)[0]
        except Exception as error:
            st.error(f"Error saat perhitungan what-if: {error}")
            return

        cert_delta = cert_salary - base_salary
        skills_delta = skills_salary - base_salary

        tabs = st.tabs(["Prediction & What-If", "Market Insights"])

        with tabs[0]:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='salary-value'>{format_currency(base_salary)}</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Prediksi gaji tahunan berdasarkan profil yang diberikan</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("What-If Scenarios")
            left, right = st.columns(2)
            left.metric(
                "Tambah 1 certification",
                format_currency(cert_salary),
                f"{format_currency(cert_delta)}",
            )
            right.metric(
                "Tambah 2 skills",
                format_currency(skills_salary),
                f"{format_currency(skills_delta)}",
            )

        with tabs[1]:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Market Benchmarking")
            market_data = load_market_data()
            plot_salary_distribution(market_data["salary"], base_salary)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Feature Importance")
            plot_feature_importance(artifacts)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Klik tombol Predict Salary untuk melihat hasil dan visualisasi.")


if __name__ == "__main__":
    render_main()
