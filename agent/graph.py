from typing import TypedDict, List, Dict, Any
from pathlib import Path
from langgraph.graph import StateGraph, END

# Værktøjer
from .tools import needs as needs_mod
from .tools import profiling as dq_mod
from .tools import modeling as mdl_mod
from .tools.reporting import build_context, render_html, html_to_pdf
from .tools.utils import new_run_dir, load_config

# ---- State ----
class AgentState(TypedDict, total=False):
    plan: List[str]               # fx ["prepare","needs","dq","model","report","pdf"]
    cursor: int                   # intern peger til næste step i plan
    interviews_dir: str           # sti til interview-tekster
    csv_path: str                 # sti til csv
    run_dir: str                  # output-mappe pr. kørsel
    cfg: Dict[str, Any]           # konfiguration
    logs: List[str]               # loglinjer til UI

    # Artefakter
    needs: Dict[str, Any]
    dq: Dict[str, Any]
    model: Dict[str, Any]
    html_path: str
    pdf_path: str
    next: str                     # routerens valg af næste node

# ---- Hjælpere ----
ALLOWED_STEPS = {"prepare","needs","dq","model","report","pdf","end"}

def _log(state: AgentState, msg: str) -> AgentState:
    logs = list(state.get("logs", []))
    logs.append(msg)
    state["logs"] = logs
    return state

def _want(state: AgentState, step: str) -> bool:
    """Returnér True hvis step er i planen (eller plan er None = kør alt)."""
    plan = state.get("plan")
    return True if plan is None else (step in plan)

def _ensure_run_dir(state: AgentState) -> Path:
    """Sørger for at der er en run-folder; opretter hvis mangler."""
    rd = state.get("run_dir")
    if not rd:
        rd = new_run_dir().as_posix()
        state["run_dir"] = rd
        _log(state, f"Auto-prepare: created run dir {rd}")
    p = Path(rd)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ensure_cfg(state: AgentState) -> Dict[str, Any]:
    """Sørger for at der er cfg i state; indlæser hvis mangler."""
    cfg = state.get("cfg")
    if cfg is None:
        cfg = load_config()
        state["cfg"] = cfg
        _log(state, "Auto-prepare: config loaded")
    return cfg

# ---- Noder (alle returnerer state/dict) ----
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
    _ensure_run_dir(state); cfg = _ensure_cfg(state)
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
    _ensure_run_dir(state); _ensure_cfg(state)
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
    run_dir = _ensure_run_dir(state); cfg = _ensure_cfg(state)
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
            run_dir=run_dir
        )
        state["model"] = res
        return _log(state, f"Model: trained target='{target}', shap={want_shap}")
    except Exception as e:
        return _log(state, f"Model: ERROR {e}")

def node_report(state: AgentState) -> AgentState:
    if not _want(state, "report"):
        return _log(state, "Skip: report")
    run_dir = _ensure_run_dir(state); cfg = _ensure_cfg(state)
    ctx = build_context(
        state.get("needs", {}), state.get("dq", {}),
        state.get("model", {}), cfg
    )
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
    _ensure_run_dir(state); _ensure_cfg(state)
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

# ---- Router (returnerer ALTID state/dict) ----
def router(state: AgentState) -> AgentState:
    plan = state.get("plan", [])
    idx = int(state.get("cursor", 0))
    # Hvis ingen plan er angivet -> kør alle i standardrækkefølge
    default_plan = ["prepare","needs","dq","model","report","pdf"]
    active_plan = default_plan if not plan else plan

    if idx >= len(active_plan):
        state["next"] = "end"
    else:
        step = active_plan[idx]
        state["cursor"] = idx + 1
        state["next"] = step if step in ALLOWED_STEPS else "end"
    return state

# ---- Byg graf ----
def build_graph():
    sg = StateGraph(AgentState)

    # Noder
    sg.add_node("router",  router)   # router returnerer state og sætter state["next"]
    sg.add_node("prepare", node_prepare)
    sg.add_node("needs",   node_needs)
    sg.add_node("dq",      node_dq)
    sg.add_node("model",   node_model)
    sg.add_node("report",  node_report)
    sg.add_node("pdf",     node_pdf)

    # Entry point
    sg.set_entry_point("router")

    # Conditional edges der læser 'next' fra state (IKKE kalder router igen)
    sg.add_conditional_edges(
        "router",
        lambda s: s.get("next", "end"),
        {
            "prepare": "prepare",
            "needs":   "needs",
            "dq":      "dq",
            "model":   "model",
            "report":  "report",
            "pdf":     "pdf",
            "end":     END,
        }
    )

    # Efter hvert step går vi tilbage til router for at vælge næste
    for n in ["prepare","needs","dq","model","report","pdf"]:
        sg.add_edge(n, "router")

    return sg.compile()

# ---- Public API ----
def run_plan(
    plan: List[str] | None = None,
    interviews_dir: str = "data/interviews",
    csv_path: str = "data/sample.csv"
) -> AgentState:
    """
    Kør workflowet. Eksempler på plan:
      - None                              -> kør alle trin i standard rækkefølge
      - ["dq"]                            -> kun datakvalitet
      - ["report","pdf"]                  -> lav rapport/PDF (run_dir og cfg oprettes automatisk)
      - ["prepare","needs","model"]       -> kør udvalgte trin
    """
    graph = build_graph()
    state: AgentState = {
        "plan": plan,
        "cursor": 0,
        "interviews_dir": interviews_dir,
        "csv_path": csv_path,
        "logs": []
    }
    final_state = graph.invoke(state)  # synkron eksekvering
    return final_state
