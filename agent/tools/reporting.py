import datetime as dt
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def render_html(
    context: dict,
    template_dir: str = "reports/templates",
    template: str = "executive_report.html.j2",
    out_path: Path | None = None,
) -> str:
    """Render a Jinja2 HTML template with the given context."""
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    html = env.get_template(template).render(**context)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
    return html


def html_to_pdf(html: str, pdf_path: Path) -> None:
    """Convert an HTML string to PDF using xhtml2pdf."""
    from xhtml2pdf import pisa  # make sure xhtml2pdf is installed

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(html, dest=f)


def build_context(needs: dict, dq: dict, model: dict, cfg: dict) -> dict:
    """Build the report context passed to the HTML template."""
    return {
        "report": {
            "title": cfg.get("report", {}).get("title", "Executive Report"),
            "client_name": cfg.get("report", {}).get("client_name", ""),
            "date": dt.date.today().isoformat(),
            "tldr": (
                "The project recommends focusing on data quality, standardized reporting, "
                "and churn-reducing initiatives."
            ),
        },
        "needs": needs,
        "delivery": {
            "phases": [
                {
                    "name": "Phase 1: Discovery",
                    "weeks": 4,
                    "outputs": ["Interviews", "Needs map", "KPIs"],
                },
                {
                    "name": "Phase 2: Data Foundation",
                    "weeks": 6,
                    "outputs": ["Data profiling", "Datasets", "Standard reports"],
                },
                {
                    "name": "Phase 3: ML Pilot",
                    "weeks": 6,
                    "outputs": ["Baseline model", "Validation", "Deployment outline"],
                },
            ]
        },
        "dq": dq,
        "model": {
            "name": model.get("name", ""),
            "validation": model.get("validation", ""),
            "metrics": model.get("metrics", []),
            "shap_path": model.get("shap_path", ""),
        },
        "recommendations": {
            "top": [
                {
                    "title": "Improve data quality",
                    "text": "Automate data checks and define clear responsibilities.",
                },
                {
                    "title": "Quick win",
                    "text": "Establish weekly standard reports with key KPIs.",
                },
                {
                    "title": "ML pilot",
                    "text": "Start with churn-related initiatives and measure the effect.",
                },
            ]
        },
    }
