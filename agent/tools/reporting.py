from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import datetime as dt
import json

def render_html(context: dict, template_dir="reports/templates", template="executive_report.html.j2", out_path:Path|None=None):
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    html = env.get_template(template).render(**context)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
    return html

def html_to_pdf(html: str, pdf_path: Path):
    from xhtml2pdf import pisa
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(html, dest=f)

def build_context(needs, dq, model, cfg:dict):
    return {
        "report": {
            "title": cfg.get("report",{}).get("title", "Executive Rapport"),
            "client_name": cfg.get("report",{}).get("client_name",""),
            "date": dt.date.today().isoformat(),
            "tldr": "Projektet anbefaler fokus på datakvalitet, standardrapporter og churn-reducerende tiltag."
        },
        "needs": needs,
        "delivery": {"phases":[
            {"name":"Fase 1: Discovery", "weeks":4, "outputs":["Interviews","Needs map","KPIer"]},
            {"name":"Fase 2: Data Foundation", "weeks":6, "outputs":["Dataprofilering","Datasæt","Standardrapporter"]},
            {"name":"Fase 3: ML Pilot", "weeks":6, "outputs":["Baseline-model","Validering","Driftsskitse"]}
        ]},
        "dq": dq,
        "model": {
            "name": model.get("name",""),
            "validation": model.get("validation",""),
            "metrics": model.get("metrics",[]),
            "shap_path": model.get("shap_path","")
        },
        "recommendations": {"top":[
            {"title":"Styrk datakvalitet", "text":"Automatiser datatjek og definer klare ansvarspunkter."},
            {"title":"Hurtig gevinst", "text":"Etabler ugentlige standardrapporter med nøgle-KPI’er."},
            {"title":"ML-pilot", "text":"Start med churn-påvirkende indsatser og mål effekt."}
        ]}
    }
