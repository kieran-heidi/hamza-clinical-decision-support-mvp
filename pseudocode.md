# Clinical Decision Support (PoC) - Data Flow Pseudocode


## 1. Input: Visit Note


- The user inputs a clinical note (demo note) into the `st.text_area` component in `app.py`.
- This note contains patient history, examination findings, assessment, and a preliminary plan.


## 2. Trigger: Generate Plan Button


- The user clicks the "Generate plan" button, initiating the data processing pipeline.


## 3. Load Prompts


- The system loads prompts from `.txt` files using the `load_prompts_versioned` function in `llm_rag_demo.py`.
- These prompts are versioned and stored in a dictionary.
- Fallback prompts are used if the `.txt` files are missing.


## 4. Plan Queries (LLM)


- The `plan_queries` function in `llm_rag_demo.py` uses the LLM to generate search queries based on the visit note.
- It uses the "PLANNER_SYSTEM" prompt to instruct the LLM to extract relevant queries.
- The LLM returns a JSON object containing a list of queries.
- Example: `{"queries": ["croup management", "prednisone dose croup", "ICD-10 J05.0"]}`


## 5. Retrieve Context (RAG)


- The `retrieve_context` function in `llm_rag_demo.py` uses the generated queries to retrieve context from the ChromaDB vector store.
- It queries the vector store using the `RAGClient` class.
- The function stitches neighboring windows of text to provide more context.
- It returns a dictionary containing the concatenated context, page numbers, and queries used.


## 6. Generate Management Plan (LLM)


- The `answer_with_context` function in `llm_rag_demo.py` uses the LLM to generate a management plan based on the retrieved context.
- It uses the "ANSWER_SYSTEM" and "ANSWER_USER_TMPL" prompts to instruct the LLM.
- The LLM generates a management plan tailored to the note, including assessment, treatment, monitoring, and disposition.
- The function also adds citations to the management plan.


## 7. Output: Management Plan


- The generated management plan is displayed in the Streamlit app using `st.markdown`.
- The plan includes citations to the relevant sections of the guideline.


## 8. Generate Dosing Table (Optional)


- If the user clicks the "Generate dosing table" button, the system generates a dosing table based on the management plan and retrieved context.
- The `generate_dosing_queries_from_plan` function uses the LLM to extract dosing-focused queries from the management plan.
- The `retrieve_dosing_context` function retrieves dosing context from the ChromaDB vector store.
- The `build_dosing_table` function uses the LLM to generate a dosing table based on the retrieved context.
- The dosing table is displayed in the Streamlit app using `st.markdown`.


## 9. Quick Dose Calculator (Optional)


- The user can use the quick dose calculator to calculate the dose of a medication based on the patient's weight and the recommended dose.
- The calculator is rendered using the `render_dose_calculator_form` function.
- The calculated dose is displayed in the Streamlit app.

