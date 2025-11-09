# app.py
import os
from pathlib import Path
import streamlit as st

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".") / ".env")
except Exception:
    pass

# Import your RAG + LLM helpers
from scripts.llm_rag_demo import (
    RAGClient,
    plan_queries,
    retrieve_context,
    answer_with_context,
    DOC_ID,
)

st.set_page_config(page_title="Heidi CDS (PoC)", layout="centered")
st.title("Heidi • Clinical Decision Support (PoC)")
st.caption("Demo only — not medical advice.")

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    doc_id = st.text_input("Document ID", DOC_ID)
    k = st.slider("Top-k", 1, 10, 4)
    above = st.number_input("Neighbors above", min_value=0, max_value=3, value=1, step=1)
    below = st.number_input("Neighbors below", min_value=0, max_value=4, value=2, step=1)
    show_context = st.checkbox("Show retrieved context", value=True)
    api_set = bool(os.getenv("OPENAI_API_KEY"))
    st.write("✅ OPENAI_API_KEY loaded" if api_set else "⚠️ OPENAI_API_KEY missing")

# Input note
demo_note = (
    "3-year-old with barky cough, hoarse voice, inspiratory stridor at rest. "
    "Temp 38°C. Mild chest retractions. No drooling. No allergies known."
)
note = st.text_area("Visit note / transcript", value=demo_note, height=200)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Generate plan", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if run_btn:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set (put it in .env or set as an env var).")
    elif not note.strip():
        st.error("Please enter a visit note.")
    else:
        rag = RAGClient(db_path="./chroma_db")

        with st.spinner("Planning queries…"):
            queries = plan_queries(note)
        st.write("**Queries:** ", ", ".join(queries))

        with st.spinner("Retrieving guideline context…"):
            ctx = retrieve_context(
                rag, queries, doc_id=doc_id, k=k, above=above, below=below
            )

        if not ctx["context"]:
            st.error(f"No context found. Tried queries: {queries}")
        else:
            if show_context:
                st.subheader("Retrieved context (stitched)")
                st.code(
                    ctx["context"][:4000] + ("..." if len(ctx["context"]) > 4000 else "")
                )
                st.caption(
                    f"Source: Respiratory chapter, p. {ctx['pages'][0]}–{ctx['pages'][1]}"
                )

            with st.spinner("Drafting management plan…"):
                plan_md = answer_with_context(note, ctx["context"], ctx["pages"])

            st.subheader("Management plan")
            st.markdown(plan_md)
            st.caption("Citations included at the end of the response.")
