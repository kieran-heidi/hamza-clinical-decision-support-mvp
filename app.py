# app.py
import os
from pathlib import Path
import streamlit as st

# ---- Env (.env at project root) ----
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".") / ".env")
except Exception:
    pass

# ---- Import your helpers ----
from scripts.llm_rag_demo import (
    RAGClient,
    plan_queries,
    retrieve_context,
    answer_with_context,
    load_prompts_versioned,
    DOC_ID,
)

# ---- Streamlit page ----
st.set_page_config(page_title="Heidi CDS (PoC)", layout="centered")
st.title("Heidi • Clinical Decision Support (PoC)")
st.caption("Demo only — not medical advice.")

DB_PATH = str(Path.cwd() / "chroma_db")

# ---- Session state init ----
def _init_state():
    for k, v in {
        "plan_ready": False,
        "plan_md": "",
        "ctx": None,
        "queries": [],
        "doc_id_used": "",
        "error": "",
    }.items():
        st.session_state.setdefault(k, v)
_init_state()

# ---- Utilities ----
@st.cache_data(show_spinner=False)
def get_doc_ids(db_path: str):
    """Return all doc_ids in the vector store (cached)."""
    try:
        rag_tmp = RAGClient(db_path=db_path)
        return rag_tmp.list_doc_ids()  # ensure RAGClient has list_doc_ids()
    except Exception:
        return []

def clear_plan_state():
    st.session_state["plan_ready"] = False
    st.session_state["plan_md"] = ""
    st.session_state["ctx"] = None
    st.session_state["queries"] = []
    st.session_state["doc_id_used"] = ""
    st.session_state["error"] = ""

def render_dose_calculator_form():
    """Quick dose calculator as a FORM; persists last result in session state."""
    st.divider()
    st.markdown("### Quick dose calculator")

    dkey = "dose_state"
    if dkey not in st.session_state:
        st.session_state[dkey] = {
            "weight": 12.0,
            "mgkg": 0.6,
            "drug": "dexamethasone",
            "use_cap": True,
            "cap": 10.0,
            "use_round": True,
            "round_inc": 0.5,
            "exact": None,
            "capped": None,
            "rounded": None,
        }

    def _round_to_inc(value: float, inc: float) -> float:
        if inc and inc > 0:
            return round(value / inc) * inc
        return value

    with st.form("dose_calc_form", clear_on_submit=False):
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            weight_kg = st.number_input(
                "Patient weight (kg)",
                min_value=2.0, max_value=80.0, step=0.5,
                value=st.session_state[dkey]["weight"],
                key="form_weight",
            )
        with colB:
            mg_per_kg = st.number_input(
                "Recommended dose (mg/kg)",
                min_value=0.05, max_value=5.0, step=0.05,
                value=st.session_state[dkey]["mgkg"],
                key="form_mgkg",
            )
        with colC:
            drug_name = st.text_input(
                "Medication (label only)",
                value=st.session_state[dkey]["drug"],
                key="form_drug",
            )

        c1, c2 = st.columns([1, 1])
        with c1:
            use_cap = st.checkbox(
                "Apply max cap (mg)?",
                value=st.session_state[dkey]["use_cap"],
                key="form_cap_on",
            )
            cap_mg = st.number_input(
                "Max dose (mg)",
                min_value=1.0, max_value=100.0, step=0.5,
                value=st.session_state[dkey]["cap"],
                disabled=not use_cap, key="form_cap",
            )
        with c2:
            use_round = st.checkbox(
                "Round to increment (mg)?",
                value=st.session_state[dkey]["use_round"],
                key="form_round_on",
            )
            round_inc = st.number_input(
                "Rounding increment (mg)",
                min_value=0.1, max_value=10.0, step=0.1,
                value=st.session_state[dkey]["round_inc"],
                disabled=not use_round, key="form_round",
            )

        submitted = st.form_submit_button("Calculate dose")

    if submitted:
        exact_mg = weight_kg * mg_per_kg
        capped_mg = min(exact_mg, cap_mg) if use_cap else exact_mg
        rounded_mg = _round_to_inc(capped_mg, round_inc) if use_round else capped_mg
        st.session_state[dkey].update({
            "weight": weight_kg, "mgkg": mg_per_kg, "drug": drug_name,
            "use_cap": use_cap, "cap": cap_mg, "use_round": use_round, "round_inc": round_inc,
            "exact": exact_mg, "capped": capped_mg, "rounded": rounded_mg,
        })

    ds = st.session_state[dkey]
    if ds["exact"] is not None:
        st.write("**Calculation**")
        lines = [
            "Weight × mg/kg = exact dose",
            f"{ds['weight']:.1f} kg × {ds['mgkg']:.3g} mg/kg = {ds['exact']:.2f} mg",
        ]
        if ds["use_cap"]:
            lines.append(f"Apply cap: min({ds['exact']:.2f}, {ds['cap']:.2f}) = {ds['capped']:.2f} mg")
        if ds["use_round"]:
            lines.append(f"Round to {ds['round_inc']:.3g} mg → {ds['rounded']:.2f} mg")
        st.code("\n".join(lines))
        st.success(f"**Administered dose: {ds['rounded']:.2f} mg** of {ds['drug']}")

        # Always-balanced detail suffix
        steps = []
        if ds["use_cap"] or ds["use_round"]:
            steps.append(f"exact {ds['exact']:.2f}")
        if ds["use_cap"]:
            steps.append(f"cap {ds['capped']:.2f}")
        if ds["use_round"]:
            steps.append(f"round {ds['round_inc']:.3g} mg")
        detail = f" ({' → '.join(steps)})" if steps else ""

        dose_block_md = (
            f"- Drug: **{ds['drug']}**\n"
            f"- Weight: **{ds['weight']:.1f} kg**\n"
            f"- Rule: **{ds['mgkg']:.3g} mg/kg**"
            + (f", cap **{ds['cap']:.3g} mg**" if ds["use_cap"] else "")
            + "\n"
            f"- Result: **{ds['rounded']:.2f} mg**{detail}"
        )
        with st.expander("Copy dose block"):
            st.markdown(dose_block_md)
    else:
        st.info("Enter values and click **Calculate dose** to see the result.")

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Settings")

    # Refresh doc list cache
    if st.button("🔄 Refresh doc list"):
        get_doc_ids.clear()

    all_doc_ids = get_doc_ids(DB_PATH)
    if not all_doc_ids:
        st.warning(f"No doc_ids found in {DB_PATH}. Make sure you've ingested a PDF.")
        doc_id = DOC_ID
    else:
        default_index = all_doc_ids.index(DOC_ID) if DOC_ID in all_doc_ids else 0
        doc_id = st.selectbox("Document ID", all_doc_ids, index=default_index)

    # Retrieval hyperparams
    k = st.slider("Top-k", 1, 10, 4)
    above = st.number_input("Neighbors above", min_value=0, max_value=3, value=1, step=1)
    below = st.number_input("Neighbors below", min_value=0, max_value=4, value=2, step=1)

    # Prompts (versioned)
    st.markdown("### Prompts")
    prompts_base = st.text_input("Prompts base dir", "./prompts")
    prompts_version = st.text_input("Version (blank = latest or 'current')", "")
    show_context = st.checkbox("Show retrieved context", value=True)

    # API
    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.write("✅ OPENAI_API_KEY loaded" if api_ok else "⚠️ OPENAI_API_KEY missing")

# ---- Main inputs ----
demo_note = (
"""Patient: Jack T.
DOB: 12/03/2022
Age: 3 years
Weight: 14.2 kg

Presenting complaint:
Jack presented with a 2-day history of barky cough, hoarse voice, and low-grade fever. Symptoms worsened overnight, with increased work of breathing and stridor noted at rest this morning. No history of choking, foreign body aspiration, or recent travel. No known sick contacts outside the household. 

History:
- Onset of URTI symptoms 2 days ago, including rhinorrhoea and dry cough
- Barking cough began yesterday evening, hoarseness and intermittent inspiratory stridor overnight
- Mild fever (up to 38.4°C) controlled with paracetamol
- No cyanosis or apnoea reported
- Fully vaccinated and developmentally appropriate for age
- No history of asthma or other chronic respiratory illness
- No previous episodes of croup
- No drug allergies

Examination:
- Alert, mildly distressed, sitting upright with audible inspiratory stridor at rest
- Barky cough noted during assessment
- Mild suprasternal and intercostal recession
- RR 32, HR 124, SpO2 97% on room air, T 37.9°C
- Chest: clear air entry bilaterally, no wheeze or crackles
- ENT: mild erythema of oropharynx, no tonsillar exudate
- CVS: normal S1/S2, no murmurs
- Neurological: alert, interactive, normal tone and reflexes

Assessment:
Jack presents with classic features of moderate croup (laryngotracheobronchitis), likely viral in origin. No signs of severe respiratory distress or impending airway obstruction. No signs suggestive of bacterial tracheitis or other differentials (e.g. foreign body, epiglottitis).

Plan:
- Administer corticosteroids
- Plan as per local guidelines for croup"""
)
note = st.text_area("Visit note / transcript", value=demo_note, height=200, key="note_input")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Generate plan", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    clear_plan_state()
    st.rerun()

# ---- Run pipeline (store results in session) ----
if run_btn:
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.error = "OPENAI_API_KEY not set (put it in .env or export it)."
        st.session_state.plan_ready = False
        st.rerun()
    if not note.strip():
        st.session_state.error = "Please enter a visit note."
        st.session_state.plan_ready = False
        st.rerun()

    try:
        prompts = load_prompts_versioned(Path(prompts_base), prompts_version or None)
    except Exception as e:
        st.session_state.error = f"Failed to load prompts: {e}"
        st.session_state.plan_ready = False
        st.rerun()

    rag = RAGClient(db_path=DB_PATH)

    with st.spinner("Planning queries…"):
        queries = plan_queries(note, prompts)
    with st.spinner("Retrieving guideline context…"):
        ctx = retrieve_context(rag, queries, doc_id=doc_id, k=k, above=above, below=below)
    if not ctx["context"]:
        st.session_state.error = f"No context found. Tried queries: {queries}"
        st.session_state.plan_ready = False
        st.rerun()

    with st.spinner("Drafting management plan…"):
        plan_md = answer_with_context(note, ctx["context"], ctx["pages"], prompts)

    # Persist
    st.session_state.plan_ready = True
    st.session_state.plan_md = plan_md
    st.session_state.ctx = ctx
    st.session_state.queries = queries
    st.session_state.doc_id_used = doc_id
    st.session_state.error = ""
    st.rerun()

# ---- Render from session (stable across reruns) ----
if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.plan_ready and st.session_state.ctx:
    ctx = st.session_state.ctx

    st.write("**Queries:** ", ", ".join(st.session_state.queries))
    if show_context:
        st.subheader("Retrieved context (stitched)")
        preview = ctx["context"][:4000] + ("..." if len(ctx["context"]) > 4000 else "")
        st.code(preview)
        st.caption(f"Source: {st.session_state.doc_id_used}, p. {ctx['pages'][0]}–{ctx['pages'][1]}")

    st.subheader("Management plan")
    st.markdown(st.session_state.plan_md)
    st.caption("Citations included at the end of the response.")

    # ---- Calculator appears BELOW the plan ----
    render_dose_calculator_form()
