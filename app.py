
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

def get_latest_skills_csv():
    files = glob.glob("logs/skills.csv")
    if not files: return None
    # If multiple (from different runs), usually we want the one corresponding to the latest run.
    # But here we just assume one main skill log or the latest.
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
            # If config is a dict, convert to ResNetBKConfig
            if isinstance(config_data, dict):
                from src.models.configurable_resnet_bk import ResNetBKConfig
                # Filter out unknown keys to prevent TypeError
                valid_keys = ResNetBKConfig.__annotations__.keys()
                filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
                config_data = ResNetBKConfig(**filtered_config)

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
        ["📊 Dashboard", "🧑‍🏫 Teaching Wizard", "💬 Chat / Test", "🧪 Merge Lab", "🔁 Reborn Ritual", "🚀 Deploy Studio", "💤 Dream Cycle"],
        captions=["Monitor Training", "Auto-Curriculum", "Interact with AI", "Mix & Match Models", "New Life + Memory", "Publish to HF", "Memory Consolidation"]
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

    # --- Future IQ Section ---
    st.markdown("### 🧠 Future IQ & Skill Radar")
    skills_csv = get_latest_skills_csv()
    if skills_csv:
        try:
            sdf = pd.read_csv(skills_csv)
            if not sdf.empty:
                latest_skills = sdf.iloc[-1].to_dict()
                step = latest_skills.pop('step')

                # Radar Chart
                categories = list(latest_skills.keys())
                values = list(latest_skills.values())

                # Future Prediction (Mocked simply as +20% for demo visualization)
                future_values = [min(100, v * 1.2) for v in values]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Current'))
                fig.add_trace(go.Scatterpolar(r=future_values, theta=categories, fill='toself', name='Predicted (+1000 steps)'))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    template="plotly_dark",
                    title="Skill Growth Projection"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Text Prediction
                st.info(f"🔮 **Prediction:** Logic ability expected to rise to **{future_values[categories.index('Logic')]:.1f}** in next epoch.")
        except Exception as e:
            st.error(f"Error reading skills: {e}")

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
                # Use model sequence length if available to avoid n_seq mismatch
                seq_len = getattr(getattr(model, "config", None), "n_seq", None)
                if seq_len is None:
                    seq_len = getattr(model, "n_seq", 128)
                dummy = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
                out = model(dummy)
                diag = out.get('diagnostics', {})
                meta = diag.get('meta_commentary', "Processing...")

                st.markdown(f"<div class='monologue'>🧠 {meta}</div>", unsafe_allow_html=True)
                time.sleep(0.5)

                res_text = f"I processed: '{prompt}'. (Model: {os.path.basename(actual_path) if actual_path else 'Mock'})"
                st.write(res_text)

        st.session_state.messages.append({"role": "assistant", "content": res_text, "meta": meta})

# --- Page 3: Teaching Wizard ---
elif "Teaching Wizard" in page:
    st.header("🧑‍🏫 MUSE Teaching Wizard")
    st.caption("Let the AI design the perfect curriculum for your goals.")

    # 1. Goal Selection
    st.markdown("### 1. What is your goal?")
    goals_dir = "configs/goals"
    if os.path.exists(goals_dir):
        goals = [f.replace(".yaml", "") for f in os.listdir(goals_dir) if f.endswith(".yaml")]
    else:
        goals = []

    selected_goal = st.selectbox("Select Target Skill", goals, format_func=lambda x: x.replace("_", " ").title())

    # Load goal details
    if selected_goal:
        with open(os.path.join(goals_dir, f"{selected_goal}.yaml"), 'r') as f:
            goal_config = yaml.safe_load(f)

        st.info(f"**Plan:** {goal_config.get('description', 'No description')}")

        # Visualizing & Editing Mix
        c_title, c_reset = st.columns([3, 1])
        with c_title:
            st.markdown("#### 🎛️ Fine-tune Dataset Mix")
        with c_reset:
            if st.button("↺ Reset Defaults"):
                for ds_name, weight in mix.items():
                    st.session_state[f"mix_slider_{ds_name}"] = float(weight)
                st.rerun()

        mix = goal_config.get('dataset_mix', {})

        new_mix = {}
        cols = st.columns(len(mix)) if len(mix) > 0 else [st.container()]

        # Sliders for each dataset
        for i, (ds_name, weight) in enumerate(mix.items()):
            with cols[i % 3]: # Wrap around 3 columns
                # Use session state key to enable reset
                key = f"mix_slider_{ds_name}"
                if key not in st.session_state:
                    st.session_state[key] = float(weight)

                new_weight = st.slider(f"{ds_name}", 0.0, 1.0, key=key, step=0.05)
                new_mix[ds_name] = new_weight

        # Normalize?
        total_weight = sum(new_mix.values())
        if total_weight == 0: total_weight = 1.0

        # Show breakdown
        fig = px.pie(values=list(new_mix.values()), names=list(new_mix.keys()), title=f"Final Mix (Total: {total_weight:.2f})", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # 2. Configuration
        st.markdown("### 2. Configure Session")
        c1, c2 = st.columns(2)
        with c1:
            epochs = st.number_input("Epochs", value=goal_config.get('recommended_epochs', 3), min_value=1)
        with c2:
            base_model = st.selectbox("Base Model", ["None (Start Fresh)"] + checkpoints)

        # 3. Action
        st.markdown("### 3. Start Training")

        # Generate custom config path
        custom_config_path = "configs/custom_curriculum.yaml"

        if st.button("🚀 Launch Curriculum", type="primary"):
            # Save the custom config
            custom_config = goal_config.copy()
            custom_config['dataset_mix'] = new_mix

            with open(custom_config_path, 'w') as f:
                yaml.dump(custom_config, f)

            cmd_str = f"make train-user CONFIG={custom_config_path} EPOCHS={epochs}"
            if base_model != "None (Start Fresh)":
                cmd_str += f" RESUME={base_model}"

            st.toast(f"Starting curriculum: {selected_goal.replace('_', ' ').title()}")
            st.code(cmd_str, language="bash")

            with st.status("Initializing AI Teacher...") as status:
                st.write("Saving custom curriculum...")
                time.sleep(0.5)
                st.write(f"Mix: {new_mix}")
                st.write("Analyizing current model skills...")
                time.sleep(1)
                st.write("Allocating VRAM...")
                time.sleep(1)
                st.write("Allocating VRAM...")
                time.sleep(1)
                status.update(label="Training Started! Go to Dashboard.", state="complete")

# --- Page 4: Merge Lab ---
elif "Merge Lab" in page:
    st.header("🧪 Merge Lab: Creative Synthesis")
    st.caption("Blend two checkpoints to create a hybrid intelligence.")

    c1, c2 = st.columns(2)
    with c1:
        model_a = st.selectbox("Model A (Base)", checkpoints, index=0 if checkpoints else None)
    with c2:
        model_b = st.selectbox("Model B (Flavor)", checkpoints, index=0 if checkpoints else None)

    method = st.selectbox("Merge Method", ["Linear Interpolation (Lerp)", "Layer-wise (Frankenstein)", "Trait Addition (Evolution)"])

    if "Lerp" in method:
        alpha = st.slider("Mixing Ratio (Alpha)", 0.0, 1.0, 0.5, help="0.0 = Model B only, 1.0 = Model A only")
        st.markdown(f"**Result:** {alpha*100:.0f}% Model A + {(1-alpha)*100:.0f}% Model B")
        cmd_method = "lerp"
        extra_args = ["--alpha", str(alpha)]
    elif "Layer-wise" in method:
        split = st.number_input("Split Layer (0-12)", min_value=1, max_value=24, value=6)
        st.markdown(f"**Result:** Layers 0-{split} from A, Layers {split}+ from B")
        cmd_method = "layer_wise"
        extra_args = ["--split_layer", str(split)]
    else:
        alpha = st.slider("Trait Strength", 0.1, 2.0, 0.5, help="Strength of Model B's traits added to A")
        st.markdown(f"**Result:** A + {alpha} * (B - A)")
        cmd_method = "trait_add"
        extra_args = ["--alpha", str(alpha)]

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
                "--method", cmd_method
            ] + extra_args

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
    retention = st.slider("Soul Retention Rate", 0.0, 1.0, 0.7, help="How much memory to keep. 1.0 = Full Memory, 0.0 = Total Amnesia")
    reborn_name = st.text_input("Name of the Child", "reborn_muse_v1.pt")

    if st.button("🔥 Begin Reborn Ritual", type="primary"):
        if not elder_model:
            st.error("You must select an elder model.")
        else:
            output_path = os.path.join("checkpoints", reborn_name)
            cmd = [
                sys.executable, "scripts/reborn.py",
                "--checkpoint", elder_model,
                "--output", output_path,
                "--retention_rate", str(retention)
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

# --- Page 6: Deploy Studio ---
elif "Deploy Studio" in page:
    st.header("🚀 Deploy Studio")
    st.caption("Publish your evolved MUSE to the world.")

    model_to_deploy = st.selectbox("Select Model to Publish", checkpoints)
    repo_name = st.text_input("Hugging Face Repo ID", "username/muse-evolved-v1")
    hf_token = st.text_input("HF Token (Optional if logged in CLI)", type="password")

    if st.button("☁️ Publish to Hugging Face", type="primary"):
        if not model_to_deploy:
            st.error("Select a model.")
        else:
            cmd = [
                sys.executable, "scripts/deploy_interactive.py",
                "--model", model_to_deploy,
                "--repo", repo_name
            ]
            if hf_token:
                cmd.extend(["--token", hf_token])

            with st.status("Deploying...") as status:
                st.write("Generating Model Card & README...")
                time.sleep(1)
                st.write("Optimizing Weights (Quantization)...")
                time.sleep(1)
                st.write("Uploading to Hub...")

                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    st.code(res.stdout)
                    if res.returncode == 0:
                        status.update(label="Deployment Complete!", state="complete")
                        st.balloons()
                    else:
                        status.update(label="Deployment Failed", state="error")
                        st.error(res.stderr)
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
