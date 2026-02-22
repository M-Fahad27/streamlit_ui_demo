# app.py
"""
Streamlit UI for ExplainCode using CodeT5.
Run: streamlit run app.py
"""

import streamlit as st
from models.codet5_model import CodeExplainer
import textwrap

st.set_page_config(page_title="ExplainCode", page_icon="💡", layout="wide")

# Sidebar / config
st.sidebar.title("ExplainCode — Settings")
model_name = st.sidebar.text_input("Model (HF id)", value="Salesforce/codet5-base")
device_choice = st.sidebar.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
max_output = st.sidebar.slider(
    "Max explanation tokens", min_value=50, max_value=512, value=180, step=10
)
num_beams = st.sidebar.slider(
    "Generation beams (num_beams)", min_value=1, max_value=8, value=4
)
explain_kind = st.sidebar.selectbox(
    "Explanation style",
    options=["detailed", "brief", "include time complexity", "short"],
    index=0,
)

st.title("💡 ExplainCode — Transformer-based Code Explainer")
st.markdown(
    "Paste a code snippet on the left, choose options, and click **Explain**. "
    "Uses a single transformer model (CodeT5 by default)."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Code")
    example_code = textwrap.dedent(
        """\
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        """
    )
    code_input = st.text_area("Paste your code here", value=example_code, height=300)
    language = st.text_input("Language (for prompt)", value="python")
    st.write("Or upload a code file (.py, .java, .cpp etc.)")
    uploaded_file = st.file_uploader(
        "Upload code file", type=["py", "java", "cpp", "js", "ts", "go", "rs", "txt"]
    )
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
            code_input = file_content
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as e:
            st.error("Could not read uploaded file.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        explain_btn = st.button("🧠 Explain")
    with col_btn2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.experimental_rerun()

with col2:
    st.subheader("Output / Explanation")
    if "explainer" not in st.session_state:
        # lazy initialize model when needed to speed up UI load
        try:
            device = None if device_choice == "auto" else device_choice
            st.session_state["explainer"] = CodeExplainer(
                model_name=model_name, device=device, max_output_tokens=max_output
            )
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.stop()

    explainer: CodeExplainer = st.session_state["explainer"]

    if explain_btn:
        if not code_input.strip():
            st.warning("Please paste or upload code to explain.")
        else:
            with st.spinner("Generating explanation (this may take ~10-60s on CPU)..."):
                try:
                    explanation = explainer.explain(
                        code=code_input,
                        language=language,
                        explain_kind=explain_kind,
                        num_beams=num_beams,
                        temperature=0.7,
                        do_sample=False,
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    explanation = ""
            if explanation:
                st.markdown("**Explanation:**")
                st.code(explanation, language="markdown")
                st.download_button(
                    "Download explanation (.txt)",
                    explanation,
                    file_name="explanation.txt",
                )
    else:
        st.info(
            "Click **Explain** to generate a natural-language explanation for the code."
        )

st.markdown("---")
st.markdown("### Tips & Notes")
st.write(
    """
- Model `Salesforce/codet5-base` will be downloaded the first time (internet required).
- Prefer GPU for faster generation (CUDA). On CPU it may take longer.
- If you need longer explanations, increase 'Max explanation tokens' in the sidebar.
- For production, consider caching the explainer object and using `torch.compile` / quantization for speed.
"""
)
