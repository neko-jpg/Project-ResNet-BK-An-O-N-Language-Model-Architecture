
import streamlit as st
import torch
import pandas as pd
import numpy as np
import yaml
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import time
import json
import subprocess
import sys
from pathlib import Path
import altair as alt

# Imports
try:
    from src.utils.mock_phase3 import MockPhase3Model
    from src.models.phase4.integrated_model import Phase4IntegratedModel
    from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
except ImportError:
    st.error("Project MUSE modules not found. Run this from the repo root.")
    st.stop()

# Page Config
st.set_page_config(
    page_title="MUSE: Phase 7 Console",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌"
)

# Custom CSS
st.markdown("""
<style>
    /* Base theme */
    .stApp {
        background-color: #111111;
        color: #DDDDDD;
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        background-color: #222222;
        border: 1px solid #444444;
    }
    /* Rounded avatars */
    .st-emotion-cache-12fmjuu.e115fcil2 {
        border-radius: 50%;
    }
    /* Monologue/Inner thought box */
    .monologue {
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        color: #00FF41; /* Matrix Green */
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #00FF41;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_latest_log_file():
    """Finds the latest JSONL log file."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return None
    files = list(log_dir.glob("train_log_*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

def load_log_data(log_file):
    """Loads and parses a JSONL log file into a DataFrame."""
    if not log_file:
        return pd.DataFrame()
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue # Skip corrupted lines
    return pd.DataFrame(data)


def get_available_checkpoints():
    files = glob.glob("checkpoints/*.pt")
    files.sort(key=os.path.getmtime, reverse=True)
    return files

@st.cache_resource
def load_checkpoint_model(ckpt_path):
    if not ckpt_path:
        return MockPhase3Model(d_model=64), "Simulation Mode (Mock)"

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        config_data = ckpt.get('config')

        if isinstance(config_data, dict):
            valid_keys = ResNetBKConfig.__annotations__.keys()
            filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
            config = ResNetBKConfig(**filtered_config)
        elif isinstance(config_data, ResNetBKConfig):
            config = config_data
        else:
            # Fallback for older checkpoints
            st.warning("No config found in checkpoint, using default.")
            config = ResNetBKConfig()

        model = ConfigurableResNetBK(config)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, f"Loaded: {os.path.basename(ckpt_path)}"
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return MockPhase3Model(d_model=64), f"Error loading {os.path.basename(ckpt_path)} (Using Mock)"


@st.cache_resource
def get_muse_model(ckpt_path):
    phase3_model, status = load_checkpoint_model(ckpt_path)

    # Simplified for chat, full Phase 4 integration can be conditional
    model = Phase4IntegratedModel(
        phase3_model=phase3_model,
        enable_meta=True,
    )
    return model, status

# --- Sidebar ---
with st.sidebar:
    st.title("🌌 MUSE Console")
    st.caption("Phase 7 Creative Evolution")

    # Model Selector
    st.markdown("### 🧠 Active Core")
    checkpoints = get_available_checkpoints()
    options = ["Simulation Mode (Mock)"] + checkpoints
    selected_ckpt = st.selectbox(
        "Select Model",
        options,
        index=0, # Default to mock
        help="Choose a trained model checkpoint to interact with."
    )
    actual_path = selected_ckpt if selected_ckpt != "Simulation Mode (Mock)" else None

    if st.button("🔄 Refresh Checkpoints"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.info("This UI provides tools for interacting with and monitoring MUSE models.")


# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Dashboard", "🛠️ Model Tools"])

# --- Tab 1: Chat ---
with tab1:
    st.header("💬 Neural Interface")

    # Load Model based on selection
    with st.spinner(f"Loading {os.path.basename(actual_path) if actual_path else 'Mock'}..."):
        model, status_msg = get_muse_model(actual_path)

    if "Mock" in status_msg:
        st.warning(f"⚠️ {status_msg}")
    else:
        st.success(f"✅ {status_msg}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            if "meta" in message:
                st.markdown(f"<div class='monologue'>🧠 {message['meta']}</div>", unsafe_allow_html=True)
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask MUSE anything..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Use model's config for sequence length if available
                seq_len = getattr(model.config, "n_seq", 128)
                dummy_input = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)

                # Get model output and diagnostics
                output = model(dummy_input)
                diagnostics = output.get('diagnostics', {})
                meta_commentary = diagnostics.get('meta_commentary', "Processing input...")

                st.markdown(f"<div class='monologue'>🧠 {meta_commentary}</div>", unsafe_allow_html=True)
                response_text = f"Simulated response to: '{prompt}'. (Model: {os.path.basename(actual_path) if actual_path else 'Mock'})"
                st.markdown(response_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "meta": meta_commentary
        })

# --- Tab 2: Dashboard ---
with tab2:
    st.header("📊 Training Dashboard")
    if st.button("Refresh Logs"):
        st.rerun()

    log_file = get_latest_log_file()
    if log_file:
        df = load_log_data(log_file)
        if not df.empty:
            st.success(f"Loaded log: {log_file.name}")

            latest_row = df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Step", int(latest_row.get('step', 0)))
            c2.metric("Loss", f"{latest_row.get('loss', 0.0):.4f}")
            c3.metric("Perplexity", f"{latest_row.get('perplexity', 0.0):.2f}")

            # Interactive Chart
            st.markdown("#### Loss & Perplexity Over Time")
            base_chart = alt.Chart(df).mark_line().encode(
                x=alt.X('step:Q', title='Step'),
            ).properties(
                title="Training Metrics"
            )

            loss_chart = base_chart.encode(y=alt.Y('loss:Q', title='Loss', scale=alt.Scale(zero=False)), tooltip=['step', 'loss'])
            ppl_chart = base_chart.encode(y=alt.Y('perplexity:Q', title='Perplexity', scale=alt.Scale(zero=False)), tooltip=['step', 'perplexity'])

            # Combine charts with independent y-axes
            combined_chart = alt.layer(
                loss_chart.mark_line(color='#00FF41'),
                ppl_chart.mark_line(color='#00BFFF')
            ).resolve_scale(
                y='independent'
            )
            st.altair_chart(combined_chart, use_container_width=True)

        else:
            st.warning("Log file is empty.")
    else:
        st.info("No training logs found in the 'logs' directory.")

# --- Tab 3: Model Tools ---
with tab3:
    st.header("🛠️ Model Management Toolkit")

    st.subheader("🧬 Model Synthesis (Merge)")
    c1, c2 = st.columns(2)
    with c1:
        model_a = st.selectbox("Model A (Base)", checkpoints, index=0 if checkpoints else None, key="merge_a")
    with c2:
        model_b = st.selectbox("Model B (Flavor)", checkpoints, index=1 if len(checkpoints) > 1 else 0, key="merge_b")

    alpha = st.slider("Mixing Ratio (Alpha)", 0.0, 1.0, 0.5, help="0.0 = Model B, 1.0 = Model A")
    output_name = st.text_input("New Model Name", f"merged_{int(time.time())}.pt")

    if st.button("Synthesize New Model", type="primary"):
        if not all([model_a, model_b, output_name]):
            st.error("Please select models and provide an output name.")
        else:
            output_path = Path("checkpoints") / output_name
            cmd = [
                sys.executable, "scripts/merge_models.py",
                "--model_a", model_a, "--model_b", model_b,
                "--output", str(output_path), "--method", "lerp",
                "--alpha", str(alpha)
            ]
            with st.spinner("Merging neural weights..."):
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    st.success(f"Successfully created {output_name}!")
                    st.code(res.stdout)
                    st.balloons()
                    # Clear cache to force reload of checkpoint list
                    st.cache_resource.clear()
                except subprocess.CalledProcessError as e:
                    st.error("Merge failed.")
                    st.code(e.stderr)
