from pathlib import Path

import pandas as pd
import streamlit as st

from agent.graph import run_plan
from agent.tools import modeling as mdl_mod, needs as needs_mod, profiling as dq_mod
from agent.tools.reporting import build_context, html_to_pdf, render_html
from agent.tools.utils import load_config

st.set_page_config(page_title="BizOps AI Agent Pro", layout="wide")
st.title("BizOps AI Agent Pro")

cfg = load_config()

st.sidebar.header("Uploads")
# file_uploader (around line ~17)
uploaded_texts = st.sidebar.file_uploader(
    "Interview-filer (.txt)",
    accept_multiple_files=True,
    type=["txt"],
)

uploaded_csv = st.sidebar.file_uploader("Datasæt (.csv)", type=["csv"])

data_dir = Path("data")
interviews_dir = data_dir / "interviews"
runs_dir = data_dir / "runs"
interviews_dir.mkdir(parents=True, exist_ok=True)
runs_dir.mkdir(parents=True, exist_ok=True)

if uploaded_texts:
    for f in uploaded_texts:
        (interviews_dir / f.name).write_bytes(f.getvalue())
if uploaded_csv:
    (data_dir / "uploaded.csv").write_bytes(uploaded_csv.getvalue())

tabs = st.tabs(["Analyze needs", "Data quality", "Modeling", "Report", "Agent"])

with tabs[0]:
    st.subheader("Analyze needs")
    clusters = cfg.get("needs", {}).get("clusters", 4)
    top_terms = cfg.get("needs", {}).get("top_terms", 8)
    needs = needs_mod.run_needs(str(interviews_dir), clusters=clusters, top_terms=top_terms)
    st.write("**Themes:**")
    for t in needs["themes"]:
        st.write(f"• {t}")
    st.write("**KPI-forslag:**")
    st.table(pd.DataFrame(needs["kpis"]))

with tabs[1]:
    st.subheader("Data quality")
    csv_path = str(data_dir / ("uploaded.csv" if uploaded_csv else "sample.csv"))
    dq = dq_mod.run_dq(csv_path)
    st.json(dq)
    try:
        df_preview = pd.read_csv(csv_path).head(50)
        st.write("Preview:")
        st.dataframe(df_preview)
    except Exception as e:
        st.error(str(e))

with tabs[2]:
    st.subheader("Modeling")
    csv_path = str(data_dir / ("uploaded.csv" if uploaded_csv else "sample.csv"))
    df = pd.read_csv(csv_path)
    target_default = cfg.get("model", {}).get("target", "churn")
    if target_default not in df.columns:
        target_default = df.columns[-1]

    target = st.selectbox(
        "Vælg target-kolonne",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(target_default),
    )

    # checkbox (around ~73)
    shap_on = st.checkbox(
        "Gem SHAP-plot (hvis klassifikation)",
        value=cfg.get("model", {}).get("shap", True),
    )

    if st.button("Train baseline"):
        try:
            from pathlib import Path

            run_dir = Path("data/runs") / "manual_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            model_info = mdl_mod.run_model(
                csv_path,
                target=target,
                test_size=cfg.get("model", {}).get("test_size", 0.2),
                random_state=cfg.get("model", {}).get("random_state", 42),
                want_shap=shap_on,
                run_dir=run_dir,
            )
            st.success("Model trænet!")
            st.json(model_info)
        except Exception as e:
            st.error(str(e))

with tabs[3]:
    st.subheader("Generate report")
    csv_path = str(data_dir / ("uploaded.csv" if uploaded_csv else "sample.csv"))
    # Build ad-hoc context
    needs = needs_mod.run_needs(
        str(interviews_dir),
        clusters=cfg.get("needs", {}).get("clusters", 4),
        top_terms=cfg.get("needs", {}).get("top_terms", 8),
    )
    dq = dq_mod.run_dq(csv_path)
    mdl = mdl_mod.run_model(
        csv_path,
        target=cfg.get("model", {}).get("target", "churn"),
        test_size=cfg.get("model", {}).get("test_size", 0.2),
        random_state=cfg.get("model", {}).get("random_state", 42),
        want_shap=cfg.get("model", {}).get("shap", True),
        run_dir=Path("data/runs") / "manual_report",
    )
    ctx = build_context(needs, dq, mdl, cfg)
    html = render_html(ctx, out_path=Path("data/runs/manual_report/executive_report.html"))
    st.success("HTML report generated under data/runs/manual_report/")
    with open("data/runs/manual_report/executive_report.html", encoding="utf-8") as fh_html:
        st.download_button(
            "Download HTML report",
            data=fh_html.read(),
            file_name="executive_report.html",
            mime="text/html",
        )

    if st.button("Also export PDF"):
        from pathlib import Path

        html_str = Path("data/runs/manual_report/executive_report.html").read_text(encoding="utf-8")
        pdf_path = Path("data/runs/manual_report/executive_report.pdf")
        html_to_pdf(html_str, pdf_path)
        st.success(f"PDF created: {pdf_path}")
        with open(pdf_path, "rb") as fh_pdf:
            st.download_button(
                "Download PDF report",
                data=fh_pdf.read(),
                file_name="executive_report.pdf",
                mime="application/pdf",
            )


with tabs[4]:
    st.subheader("Agent (LangGraph)")
    st.write("Vælg plan (rækkefølge af steps) og kør agenten.")

    default_steps = ["prepare", "needs", "dq", "model", "report", "pdf"]
    step_options = ["prepare", "needs", "dq", "model", "report", "pdf"]
    selected = st.multiselect(
        "Vælg steps i rækkefølge",
        options=step_options,
        default=default_steps,
    )

    csv_path = str(data_dir / ("uploaded.csv" if uploaded_csv else "sample.csv"))
    interviews_path = str(interviews_dir)

    if st.button("Run agent plan"):
        try:
            result = run_plan(selected, interviews_dir=interviews_path, csv_path=csv_path)
            st.success("Agent run completed.")
            st.write("**Logs:**")
            for line in result.get("logs", []):
                st.write("• " + line)

            if "html_path" in result:
                from pathlib import Path

                p = Path(result["html_path"])
                if p.exists():
                    st.info(f"HTML: {p}")
                    st.download_button(
                        "Download HTML report",
                        data=p.read_text(encoding="utf-8"),
                        file_name="executive_report.html",
                        mime="text/html",
                    )

            if "pdf_path" in result:
                from pathlib import Path

                p = Path(result["pdf_path"])
                if p.exists():
                    st.info(f"PDF: {p}")
                    st.download_button(
                        "Download PDF report",
                        data=p.read_bytes(),
                        file_name="executive_report.pdf",
                        mime="application/pdf",
                    )
        except Exception as e:
            st.error(str(e))
