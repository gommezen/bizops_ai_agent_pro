# BizOps AI Agent Pro

En komplet demo-app til behovsanalyse, datakvalitet, baseline-model, og rapporter (HTML + PDF) med **run-folders** og **LangGraph-agent**.

## Hurtig start (Windows PowerShell)
```powershell
cd "C:\Users\ITSMARTSOLUTIONS\Documents\Python Scripts\bizops_ai_agent_pro"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
```

> PDF-eksport bruger **xhtml2pdf** (færre systemkrav end WeasyPrint). Hvis du har WeasyPrint, kan du skifte i `agent/tools/reporting.py`.

## Features
- **Analyze needs** (TF-IDF + KMeans) med dansk stopordsliste
- **Data quality** (missing %, duplicates, unikke værdier, simple outlier-estimat)
- **Modeling** (klassifikation/regression) + valgfri **SHAP**-plot
- **Reports** (HTML + PDF via xhtml2pdf)
- **Run-folders** (alt output gemmes under `data/runs/<timestamp>/`)
- **Agent (LangGraph)** orkestrerer needs → dq → model → report → pdf
- **Config** via `configs/config.yaml`

## Struktur
```
bizops_ai_agent_pro/
  app/main.py
  agent/graph.py
  agent/tools/{needs,profiling,modeling,reporting,utils}.py
  configs/config.yaml
  data/interviews/
  data/runs/
  reports/templates/executive_report.html.j2
  requirements.txt
  README.md
```
