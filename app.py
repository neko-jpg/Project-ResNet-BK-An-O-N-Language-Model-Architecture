
import streamlit as st
import torch
import pandas as pd
import numpy as np
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

# Imports
try:
    from src.utils.mock_phase3 import MockPhase3Model
    from src.models.phase4.integrated_model import Phase4IntegratedModel
    from src.models.configurable_resnet_bk import ConfigurableResNetBK
except ImportError:
    st.error("Project MUSE modules not found. Run this from repo root.")

# Page Config
st.set_page_config(
    page_title="MUSE: Creative Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card { background-color: #1e2127; padding: 15px; border-radius: 10px; border: 1px solid #30333d; }
    .monologue { font-family: 'Courier New', monospace; font-size: 0.8em; color: #00ff00; background-color: #000000; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 2px solid #00ff00; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def predict_future_loss(df, steps_ahead=500):
    """
    Predict future loss using Power Law Decay: L(t) = a * t^(-b) + c
    """
    if len(df) < 10: return None

    # Use steps and loss
    t = df['step'].values
    l = df['loss'].values

    # Simple Power Law: log(L) = -b*log(t) + log(a)
    # We focus on the stable regime (last 80%) to avoid warmup noise
    start_idx = int(len(t) * 0.2)
    t_fit = t[start_idx:]
    l_fit = l[start_idx:]

    if len(t_fit) < 5: return None

    try:
        # Fit linear in log-log
        # log_l = -b * log_t + log_a
        log_t = np.log(t_fit)
        log_l = np.log(l_fit)

        coeffs = np.polyfit(log_t, log_l, 1)
        b = -coeffs[0]
        a = np.exp(coeffs[1])

        # Predict
        last_step = t[-1]
        future_steps = np.arange(last_step + 1, last_step + steps_ahead + 1)
        future_loss = a * np.power(future_steps, -b)

        return pd.DataFrame({'step': future_steps, 'loss': future_loss, 'type': 'Forecast'})
    except:
        return None

def get_latest_log_csv():
    files = glob.glob("logs/*_metrics.csv")
    if not files: return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

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
        if 'config' in ckpt:
            config_data = ckpt['config']
            phase3 = ConfigurableResNetBK(config_data)
            phase3.model.load_state_dict(ckpt['model_state_dict'])
            phase3.eval()
            return phase3, f"Loaded: {os.path.basename(ckpt_path)}"
    except Exception as e:
        return MockPhase3Model(d_model=64), f"Error loading {os.path.basename(ckpt_path)} (Using Mock)"

    return MockPhase3Model(d_model=64), "Unknown Error (Using Mock)"

@st.cache_resource
def get_muse_model(ckpt_path):
    phase3, status = load_checkpoint_model(ckpt_path)

    model = Phase4IntegratedModel(
        phase3_model=phase3,
        enable_emotion=True,
        enable_dream=True,
        enable_holographic=True,
        enable_quantum=True,
        enable_topological=True,
        enable_ethics=True,
        enable_meta=True,
        enable_boundary=True
    )
    return model, status

# --- Sidebar ---
with st.sidebar:
    st.title("🧬 MUSE Studio")
    st.caption("Creative Evolution Platform")

    # Model Selector
    st.markdown("### 🧠 Active Core")
    checkpoints = get_available_checkpoints()

    # Default to first one or None
    options = ["Simulation Mode (Mock)"] + checkpoints
    selected_ckpt = st.selectbox("Select Model", options, index=1 if checkpoints else 0)

    actual_path = selected_ckpt if selected_ckpt != "Simulation Mode (Mock)" else None

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "💬 Chat / Test", "🧪 Merge Lab", "🔁 Reborn Ritual", "💤 Dream Cycle"],
        captions=["Monitor Training", "Interact with AI", "Mix & Match Models", "New Life + Memory", "Memory Consolidation"]
    )

# --- Page 1: Dashboard ---
if "Dashboard" in page:
    st.header("📊 Evolution Dashboard")
    if st.button("Refresh"): st.rerun()

    csv_path = get_latest_log_csv()
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            latest = df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Step", int(latest['step']))
            c2.metric("Loss", f"{latest['loss']:.4f}")
            c3.metric("Perplexity", f"{latest['perplexity']:.2f}")

            tab1, tab2, tab3 = st.tabs(["Loss", "Perplexity", "🔮 Future Forecast"])

            with tab1:
                st.plotly_chart(px.line(df, x="step", y="loss", title="Training Loss", template="plotly_dark"), use_container_width=True)

            with tab2:
                st.plotly_chart(px.line(df, x="step", y="perplexity", title="Perplexity", template="plotly_dark"), use_container_width=True)

            with tab3:
                forecast_df = predict_future_loss(df, steps_ahead=1000)
                if forecast_df is not None:
                    # Combine historical and forecast
                    hist_df = df[['step', 'loss']].copy()
                    hist_df['type'] = 'History'

                    combined = pd.concat([hist_df, forecast_df])

                    fig = px.line(combined, x="step", y="loss", color='type',
                                  title="Loss Convergence Prediction (Power Law)",
                                  color_discrete_map={"History": "#00ff00", "Forecast": "#ff00ff"},
                                  template="plotly_dark")
                    fig.update_traces(param=dict(dash='dash'), selector=dict(name='Forecast'))
                    st.plotly_chart(fig, use_container_width=True)

                    final_pred = forecast_df.iloc[-1]['loss']
                    st.caption(f"🔮 Predicted Loss in 1000 steps: **{final_pred:.4f}**")
                else:
                    st.info("Not enough data to forecast yet.")
        except Exception as e:
            st.error(f"Error reading log: {e}")
    else:
        st.info("No logs found.")

# --- Page 2: Chat ---
elif "Chat" in page:
    st.header("💬 Neural Interface")

    # Load Model based on selection
    with st.spinner(f"Loading {os.path.basename(actual_path) if actual_path else 'Mock'}..."):
        model, status_msg = get_muse_model(actual_path)

    if "Mock" in status_msg:
        st.warning(f"⚠️ {status_msg}")
    else:
        st.success(f"✅ {status_msg}")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Online."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "meta" in msg: st.markdown(f"<div class='monologue'>🧠 {msg['meta']}</div>", unsafe_allow_html=True)
            st.write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Mock inference for demo
                dummy = torch.randint(0, 1000, (1, 10))
                out = model(dummy)
                diag = out.get('diagnostics', {})
                meta = diag.get('meta_commentary', "Processing...")

                st.markdown(f"<div class='monologue'>🧠 {meta}</div>", unsafe_allow_html=True)
                time.sleep(0.5)

                res_text = f"I processed: '{prompt}'. (Model: {os.path.basename(actual_path) if actual_path else 'Mock'})"
                st.write(res_text)

        st.session_state.messages.append({"role": "assistant", "content": res_text, "meta": meta})

# --- Page 3: Merge Lab ---
elif "Merge Lab" in page:
    st.header("🧪 Merge Lab: Creative Synthesis")
    st.caption("Blend two checkpoints to create a hybrid intelligence.")

    c1, c2 = st.columns(2)
    with c1:
        model_a = st.selectbox("Model A (Base)", checkpoints, index=0 if checkpoints else None)
    with c2:
        model_b = st.selectbox("Model B (Flavor)", checkpoints, index=0 if checkpoints else None)

    alpha = st.slider("Mixing Ratio (Alpha)", 0.0, 1.0, 0.5, help="0.0 = Model B only, 1.0 = Model A only")

    st.markdown(f"**Result:** {alpha*100:.0f}% Model A + {(1-alpha)*100:.0f}% Model B")

    output_name = st.text_input("New Model Name", "merged_hybrid_v1.pt")

    if st.button("🧬 Synthesize New Model", type="primary"):
        if not model_a or not model_b:
            st.error("Please select two models.")
        else:
            output_path = os.path.join("checkpoints", output_name)
            cmd = [
                sys.executable, "scripts/merge_models.py",
                "--model_a", model_a,
                "--model_b", model_b,
                "--output", output_path,
                "--alpha", str(alpha)
            ]

            with st.status("Merging Neural Weights...") as status:
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0:
                        st.write(res.stdout)
                        status.update(label="Merge Complete!", state="complete")
                        st.success(f"Created {output_name}! Go to Chat to test it.")
                        # Trigger reload of checkpoints
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Merge Failed.")
                        st.code(res.stderr)
                        status.update(label="Failed", state="error")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

# --- Page 4: Reborn ---
elif "Reborn" in page:
    st.header("🔁 Reborn Ritual")
    st.caption("The Phoenix Protocol: Begin a new life while retaining ancient wisdom.")

    elder_model = st.selectbox("Select Elder Model (Source of Soul)", checkpoints)
    reborn_name = st.text_input("Name of the Child", "reborn_muse_v1.pt")

    if st.button("🔥 Begin Reborn Ritual", type="primary"):
        if not elder_model:
            st.error("You must select an elder model.")
        else:
            output_path = os.path.join("checkpoints", reborn_name)
            cmd = [
                sys.executable, "scripts/reborn.py",
                "--checkpoint", elder_model,
                "--output", output_path
            ]

            with st.status("Performing Ritual...") as status:
                st.write("1. Extracting Topological Soul (Embeddings)...")
                time.sleep(1)
                st.write("2. Forging new Neural Vessel (Random Init)...")
                time.sleep(1)
                st.write("3. Transmigrating Soul...")

                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0:
                        st.write(res.stdout)
                        status.update(label="Ritual Complete. A new MUSE is born.", state="complete")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Ritual Failed.")
                        st.code(res.stderr)
                        status.update(label="Failed", state="error")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Page 5: Sleep ---
elif "Dream" in page:
    st.header("💤 Dream Cycle")
    if st.button("Enter Sleep Mode"):
        with st.status("Dreaming..."):
            model, _ = get_muse_model(actual_path)
            model.enter_idle_mode(0.1)
            time.sleep(2)
            model.exit_idle_mode()
            st.success("Memories consolidated.")
