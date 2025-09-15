# BizOps AI Agent Pro

# Formål
Et AI-drevet analyseværktøj, der automatiserer behandlingen af tekst- og datasæt for at identificere mønstre, udlede forretningsbehov og generere KPI-forslag. Leverer komplette analyser af behov, datakvalitet, baseline-modeller og rapporter (HTML/PDF), styret via en LangGraph-agent og gemt i reproducerbare run-folders.

En komplet demo-app til behovsanalyse, datakvalitet, baseline-model, og rapporter (HTML + PDF) med **run-folders** og **LangGraph-agent**.


## Hurtig start (Windows PowerShell)

# Gå til projektmappen
cd bizops_ai_agent_pro

# Opret og aktivér virtuelt miljø
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installer afhængigheder
pip install -r requirements.txt

# Start appen
streamlit run app/main.py


## Features

**Needs analysis** → TF-IDF + KMeans, inkl. dansk stopordsliste og polering til meningsfulde temaer

**Data quality** → manglende værdier, duplicates, unikke værdier, simple outlier-estimat

**Modeling** → klassifikation/regression + valgfrit SHAP-plot for explainability

**Reporting** → HTML + PDF med Jinja2-skabelon

**Run-folders** → al output gemmes under data/runs/<timestamp>/

**LangGraph agent** → orkestrerer hele flowet (needs → dq → model → report → pdf)

**Configurable** → central styring via configs/config.yaml


## Struktur

bizops_ai_agent_pro/
├─ .github/workflows/ci.yml         # CI/CD (ruff, black, mypy)
├─ agent/                           # Orkestrering og værktøjer
│   ├─ graph.py
│   └─ tools/{needs,profiling,modeling,reporting,utils}.py
├─ app/main.py                      # Streamlit UI
├─ configs/config.yaml              # Projektkonfiguration
├─ data/                            # Input- og outputdata
│   ├─ interviews/                  # Uploadede interviews
│   └─ runs/                        # Auto-genererede analyserapporter
├─ reports/templates/executive_report.html.j2
├─ tests/                           # (WIP) unit/integration tests
├─ requirements.txt
├─ README.md
└─ pyproject.toml                   # Formatter/linter/type-check
