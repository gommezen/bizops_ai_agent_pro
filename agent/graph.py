from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

# Local imports
from .tools import modeling as mdl_mod, needs as needs_mod, profiling as dq_mod
from .tools.reporting import build_context, html_to_pdf, render_html
from .tools.utils import load_config, new_run_dir


# -----------------------------
# Agent state (shared pipeline state)
# -----------------------------
class AgentState(TypedDict, total=False):
    # Plan of steps, e.g. ["prepare","needs","dq","model","report","pdf"]
    plan: list[str] | None
    # Internal pointer to the next step in the plan
    cursor: int
    # Path to interview text files
    interviews_dir: str
    # Path to CSV dataset
    csv_path: str
    # Output directory for the current run
    run_dir: str
    # Loaded configuration
    cfg: dict[str, Any]
    # Log lines to surface in the UI
    logs: list[str]

    # Outputs (populated as the pipeline runs)
    needs: dict[str, Any]
    dq: dict[str, Any]
    model: dict[str, Any]
    html_path: str
    pdf_path: str
    # Router’s decision for the next node
    next: str


ALLOWED_STEPS = {"prepare", "needs", "dq", "model", "report", "pdf", "end"}


def _log(state: AgentState, msg: str) -> AgentState:
    """Append a message to state['logs'] and return the updated state."""
    logs = list(state.get("logs", []))
    logs.append(msg)
    state["logs"] = logs
    return state


def _want(state: AgentState, step: str) -> bool:
    """Return True if the step is in the plan (or plan is None => run everything)."""
    plan = state.get("plan")
    return True if plan is None else (step in plan)


def _ensure_run_dir(state: AgentState) -> Path:
    """Ensure a run directory exists; create it if missing and update state['run_dir']."""
    rd = state.get("run_dir")
    if not rd:
        rd = new_run_dir().as_posix()
        state["run_dir"] = rd
        _log(state, f"Auto-prepare: created run dir {rd}")
    p = Path(rd)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_cfg(state: AgentState) -> dict[str, Any]:
    """Ensure configuration is available in state; load it if missing."""
    cfg = state.get("cfg")
    if cfg is None:
        cfg = load_config()
        state["cfg"] = cfg
        _log(state, "Auto-prepare: config loaded")
    return cfg


# -----------------------------
# Nodes (each returns the updated state)
# -----------------------------
def node_prepare(state: AgentState) -> AgentState:
    if not _want(state, "prepare"):
        return _log(state, "Skip: prepare")
    run_dir = new_run_dir().as_posix()
    state["run_dir"] = run_dir
    state["cfg"] = load_config()
    return _log(state, f"Prepare: run initialized at {run_dir}")


def node_needs(state: AgentState) -> AgentState:
    if not _want(state, "needs"):
        return _log(state, "Skip: needs")
    _ensure_run_dir(state)
    cfg = _ensure_cfg(state)
    folder = state.get("interviews_dir", "data/interviews")
    clusters = cfg.get("needs", {}).get("clusters", 4)
    top_terms = cfg.get("needs", {}).get("top_terms", 8)
    try:
        res = needs_mod.run_needs(folder, clusters=clusters, top_terms=top_terms)
        state["needs"] = res
        return _log(state, f"Needs: analyzed {folder} (k={clusters}, top_terms={top_terms})")
    except Exception as e:
        return _log(state, f"Needs: ERROR {e}")


def node_dq(state: AgentState) -> AgentState:
    if not _want(state, "dq"):
        return _log(state, "Skip: dq")
    _ensure_run_dir(state)
    _ensure_cfg(state)
    csv_path = state.get("csv_path", "data/sample.csv")
    try:
        res = dq_mod.run_dq(csv_path)
        state["dq"] = res
        return _log(state, f"DQ: profiled {csv_path}")
    except Exception as e:
        return _log(state, f"DQ: ERROR {e}")


def node_model(state: AgentState) -> AgentState:
    if not _want(state, "model"):
        return _log(state, "Skip: model")
    run_dir = _ensure_run_dir(state)
    cfg = _ensure_cfg(state)
    csv_path = state.get("csv_path", "data/sample.csv")
    target = cfg.get("model", {}).get("target", "churn")
    want_shap = bool(cfg.get("model", {}).get("shap", True))
    try:
        res = mdl_mod.run_model(
            csv_path,
            target=target,
            test_size=cfg.get("model", {}).get("test_size", 0.2),
            random_state=cfg.get("model", {}).get("random_state", 42),
            want_shap=want_shap,
            run_dir=run_dir,
        )
        state["model"] = res
        return _log(state, f"Model: trained target='{target}', shap={want_shap}")
    except Exception as e:
        return _log(state, f"Model: ERROR {e}")


def node_report(state: AgentState) -> AgentState:
    if not _want(state, "report"):
        return _log(state, "Skip: report")
    run_dir = _ensure_run_dir(state)
    cfg = _ensure_cfg(state)
    ctx = build_context(state.get("needs", {}), state.get("dq", {}), state.get("model", {}), cfg)
    html_path = run_dir / "executive_report.html"
    try:
        render_html(ctx, out_path=html_path)
        state["html_path"] = html_path.as_posix()
        return _log(state, f"Report: HTML -> {html_path}")
    except Exception as e:
        return _log(state, f"Report: ERROR {e}")


def node_pdf(state: AgentState) -> AgentState:
    if not _want(state, "pdf"):
        return _log(state, "Skip: pdf")
    _ensure_run_dir(state)
    _ensure_cfg(state)
    html_file = Path(state.get("html_path", ""))
    if not html_file.exists():
        return _log(state, "PDF: no HTML found, skipping")
    pdf_path = Path(state["run_dir"]) / "executive_report.pdf"
    try:
        html = html_file.read_text(encoding="utf-8")
        html_to_pdf(html, pdf_path)
        state["pdf_path"] = pdf_path.as_posix()
        return _log(state, f"PDF: written -> {pdf_path}")
    except Exception as e:
        return _log(state, f"PDF: ERROR {e}")


# -----------------------------
# Router (ALWAYS returns the state)
# -----------------------------
def router(state: AgentState) -> AgentState:
    plan = state.get("plan", [])
    idx = int(state.get("cursor", 0))

    # If no plan is provided -> run all steps in the default order
    default_plan = ["prepare", "needs", "dq", "model", "report", "pdf"]
    active_plan = default_plan if not plan else plan

    if idx >= len(active_plan):
        state["next"] = "end"
    else:
        step = active_plan[idx]
        state["cursor"] = idx + 1
        state["next"] = step if step in ALLOWED_STEPS else "end"
    return state


# -----------------------------
# Build graph
# -----------------------------
def build_graph():
    sg = StateGraph(AgentState)

    # Nodes
    sg.add_node("router", router)  # router returns state and sets state["next"]
    sg.add_node("prepare", node_prepare)
    sg.add_node("needs", node_needs)
    sg.add_node("dq", node_dq)
    sg.add_node("model", node_model)
    sg.add_node("report", node_report)
    sg.add_node("pdf", node_pdf)

    # Entry point
    sg.set_entry_point("router")

    # Conditional edges that read 'next' from state (DO NOT call router again)
    sg.add_conditional_edges(
        "router",
        lambda s: s.get("next", "end"),
        {
            "prepare": "prepare",
            "needs": "needs",
            "dq": "dq",
            "model": "model",
            "report": "report",
            "pdf": "pdf",
            "end": END,
        },
    )

    # After each step, return to the router to choose the next node
    for n in ["prepare", "needs", "dq", "model", "report", "pdf"]:
        sg.add_edge(n, "router")

    return sg.compile()


# -----------------------------
# Public API
# -----------------------------
def run_plan(
    plan: list[str] | None = None,
    interviews_dir: str = "data/interviews",
    csv_path: str = "data/sample.csv",
) -> AgentState:
    """
    Run the workflow. Example plans:
      - None                        -> run all steps in default order
      - ["dq"]                      -> data quality only
      - ["report","pdf"]            -> generate report/PDF (run_dir and cfg are auto-prepared)
      - ["prepare","needs","model"] -> run a selected subset of steps
    """
    graph = build_graph()
    state: AgentState = {
        "plan": plan,
        "cursor": 0,
        "interviews_dir": interviews_dir,
        "csv_path": csv_path,
        "logs": [],
    }
    # Synchronous execution
    final_state = graph.invoke(state)
    return final_state
