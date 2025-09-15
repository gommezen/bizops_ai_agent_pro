### BizOps AI Agent


## Purpose
An AI-powered analysis tool that automates the processing of text and tabular datasets to identify patterns, extract business needs, and generate KPI suggestions.
It delivers complete analyses of needs, data quality, baseline models, and reports (HTML/PDF), orchestrated via a LangGraph agent and stored in reproducible run-folders.

This is a full demo application for needs analysis, data quality profiling, baseline modeling, and reporting (HTML + PDF) with run-folders and a LangGraph agent.


## Quick start (Windows PowerShell)
# Navigate to the project folder
cd bizops_ai_agent_pro


# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app/main.py

> by default the app will fall back to **data/big_sample.csv** (~ 2000 rows) if no dataset is uploaded. This to ensure meaningful demo output.


## Features
**Needs analysis** → TF-IDF + KMeans, including Danish stopword list and polishing into meaningful business themes

**Data quality** → missing values, duplicates, unique counts, simple outlier estimate

**Modeling** → classification/regression + optional SHAP plots for explainability

**Reporting** → HTML + PDF export with Jinja2 templates

**Run-folders** → all outputs stored under data/runs/<timestamp>/

**LangGraph agent** → orchestrates the full flow (needs → dq → model → report → pdf)

**Configurable** → central settings in configs/config.yaml



## Project structure
bizops_ai_agent_pro/
├─ .github/workflows/ci.yml         # CI/CD (ruff, black, mypy)
├─ agent/                           # Orchestration and tools
│   ├─ graph.py
│   └─ tools/{needs,profiling,modeling,reporting,utils}.py
├─ app/main.py                      # Streamlit UI
├─ configs/config.yaml              # Project configuration
├─ data/                            # Input and output data
│   ├─ interviews/                  # Uploaded interview files
│   └─ runs/                        # Auto-generated analysis reports
├─ reports/templates/executive_report.html.j2
├─ tests/                           # (WIP) unit/integration tests
├─ requirements.txt
├─ README.md
└─ pyproject.toml                   # Formatter/linter/type-check
