import streamlit as st
import os
import tempfile
import json
from PIL import Image

# Import your SEA modules
from env_loader import load_project_env
from pipeline import run_pipeline
from visualize import annotate_image
from langchain_openai import ChatOpenAI

# --- Page Config ---
st.set_page_config(page_title="Spatial Evidence Agent", page_icon="👁️", layout="wide")
st.title("👁️ Spatial Evidence Agent (SEA)")
st.markdown("Upload an image and ask a binary spatial question to verify relationships using geometric logic.")

# Load environment variables just in case
load_project_env()

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
backend_choice = st.sidebar.radio(
    "Choose Backend", 
    ["Local (Ollama CPU)", "OpenAI (API Key required)"]
)

# Set up the configurations based on the user's choice
if backend_choice == "Local (Ollama CPU)":
    st.sidebar.success("Using local Ollama server on CPU.")
    llm_model = "llava" # Ollama will ignore this, but LangChain needs a string
    api_key = "ollama"
    base_url = "http://localhost:11434/v1"
    vision_model = "llava"
else:
    st.sidebar.warning("Requires an active OpenAI billing balance.")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    base_url = "https://api.openai.com/v1"
    llm_model = "gpt-4o-mini"
    vision_model = "gpt-4o"

max_iterations = st.sidebar.slider("Max Correction Iterations (k)", min_value=1, max_value=3, value=3)

# --- Main UI ---
uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
question = st.text_input("Spatial Question", placeholder="e.g., Is the yellow cup to the right of the blue bottle?")

if st.button("🔍 Analyze Image", type="primary"):
    if not uploaded_file:
        st.error("Please upload an image first!")
    elif not question:
        st.error("Please ask a spatial question!")
    elif backend_choice == "OpenAI" and not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Analyzing spatial evidence... This may take a moment on CPU."):
            # 1. Save the uploaded file to a temporary location so the pipeline can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_image_path = tmp_file.name

            try:
                # 2. Setup the LLM Planner
                llm = ChatOpenAI(
                    model=llm_model,
                    api_key=api_key,
                    base_url=base_url if backend_choice == "Local (Ollama CPU)" else None,
                    temperature=0,
                    max_retries=0,
                    timeout=60,
                )

                # 3. Setup the Executor Configuration
                exec_cfg = {
                    "backend": "openai", # We use 'openai' backend logic for both real OpenAI and Ollama
                    "openai_key": api_key,
                    "openai_base": base_url if backend_choice == "Local (Ollama CPU)" else None,
                    "model": vision_model,
                }

                # 4. Run the Pipeline!
                graph = run_pipeline(
                    image_path=temp_image_path,
                    question=question,
                    llm=llm,
                    executor_config=exec_cfg,
                    critic_config={"allow_mock_models": False},
                    max_iterations=max_iterations,
                    strict_models=True,
                )

                # 5. Generate the Annotated Image
                annotated_img_path = temp_image_path.replace(".jpg", "_annotated.jpg")
                annotate_image(temp_image_path, graph, out_path=annotated_img_path)

                # --- Display Results ---
                st.divider()
                
                # Create two side-by-side columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼️ Annotated Output")
                    st.image(annotated_img_path, caption=f"Verdict: {graph.answer_str.upper()}", use_container_width=True)
                
                with col2:
                    st.subheader("📊 Spatial Evidence Graph")
                    
                    # Color code the final verdict
                    if graph.verified:
                        st.success(f"**Verified Answer:** {graph.answer_str.upper()} (Confidence: {graph.confidence})")
                    else:
                        st.error(f"**Failed to Verify:** {graph.failure_mode}")
                        
                    # Display the raw JSON evidence nicely
                    st.json(graph.model_dump())

            except Exception as e:
                st.error(f"Pipeline Error: {e}")
            
            finally:
                # Cleanup temporary files to save space
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if 'annotated_img_path' in locals() and os.path.exists(annotated_img_path):
                    os.remove(annotated_img_path)