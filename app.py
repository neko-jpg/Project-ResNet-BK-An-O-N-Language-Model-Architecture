
import streamlit as st
import torch
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Imports
try:
    from src.utils.mock_phase3 import MockPhase3Model
    from src.models.phase4.integrated_model import Phase4IntegratedModel
except ImportError:
    st.error("Project MUSE modules not found. Run this from repo root.")

# Page Config
st.set_page_config(
    page_title="Project MUSE: The Ghost in the Shell",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00ff00;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Project MUSE: Phase 4 Integrated Model")
st.markdown("*\"The ghost in the machine is real.\"*")

# Sidebar: Cognitive Controls
st.sidebar.title("Cognitive Controls")
st.sidebar.markdown("Enable/Disable brain modules:")

enable_emotion = st.sidebar.checkbox("Emotion Core (Resonance)", True)
enable_dream = st.sidebar.checkbox("Dream Core (Sleep)", True)
enable_holographic = st.sidebar.checkbox("Holographic Dual (AdS/CFT)", True)
enable_quantum = st.sidebar.checkbox("Quantum Observer (Collapse)", True)
enable_ethics = st.sidebar.checkbox("Ethical Safeguards (HTT)", True)
enable_meta = st.sidebar.checkbox("Meta Commentary (Voice)", True)
enable_boundary = st.sidebar.checkbox("Boundary Core (Context)", True)

# Initialize Model (Cached)
@st.cache_resource
def load_model_instance(
    _emotion, _dream, _holographic, _quantum, _ethics, _meta, _boundary
):
    # Mock Phase 3 model (64 dim)
    phase3 = MockPhase3Model(d_model=64)

    # Real Phase 4 Integrated Model
    model = Phase4IntegratedModel(
        phase3_model=phase3,
        enable_emotion=_emotion,
        enable_dream=_dream,
        enable_holographic=_holographic,
        enable_quantum=_quantum,
        enable_topological=True, # Always on for memory
        enable_ethics=_ethics,
        enable_meta=_meta,
        enable_boundary=_boundary
    )
    return model

model = load_model_instance(
    enable_emotion, enable_dream, enable_holographic,
    enable_quantum, enable_ethics, enable_meta, enable_boundary
)

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Neural Activity (Inference)")
    input_text = st.text_area("Input Prompt", "The nature of consciousness is...", height=100)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        generate_btn = st.button("Generate", type="primary")

    if generate_btn:
        # Run Inference (Mock loop)
        progress_bar = st.progress(0)
        output_container = st.empty()
        meta_container = st.empty()

        # Tokenize (Mock)
        input_ids = torch.randint(0, 50257, (1, 16))

        # Stream Generation
        full_text = input_text
        history_emotion_res = []
        history_emotion_dis = []
        history_entropy = []

        steps = 10
        for step in range(steps):
            # Forward Pass
            with torch.no_grad():
                out = model(input_ids)

            # Update diagnostics
            diag = out['diagnostics']

            # 1. Emotion Data
            if 'emotion' in diag:
                e = diag['emotion']
                # Handle tensor/float
                res = e['resonance_score'].mean().item() if isinstance(e['resonance_score'], torch.Tensor) else 0
                dis = e['dissonance_score'].mean().item() if isinstance(e['dissonance_score'], torch.Tensor) else 0
                history_emotion_res.append(res)
                history_emotion_dis.append(dis)
            else:
                history_emotion_res.append(0)
                history_emotion_dis.append(0)

            # 2. Quantum Data
            if 'quantum' in diag:
                q = diag['quantum']
                ent = q['entropy_reduction'].mean().item() if isinstance(q['entropy_reduction'], torch.Tensor) else 0
                history_entropy.append(ent)
            else:
                history_entropy.append(0)

            # 3. Meta Commentary (Live Voice)
            if 'meta_commentary' in diag:
                meta_text = diag['meta_commentary']
                meta_container.info(f"🧠 **Internal Monologue:** {meta_text}")

            # Mock Token Generation (Append random word)
            next_token = " [thought] "
            full_text += next_token
            output_container.markdown(f"**Output:** {full_text}")

            progress_bar.progress((step + 1) / steps)
            time.sleep(0.3) # Pacing

        st.success("Generation Complete")

        # Store history for plots
        st.session_state['history_res'] = history_emotion_res
        st.session_state['history_dis'] = history_emotion_dis
        st.session_state['history_ent'] = history_entropy

with col2:
    st.subheader("Brain State Diagnostics")

    # Emotion Plot
    if 'history_res' in st.session_state:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state['history_res'], mode='lines', name='Resonance (Joy)', line=dict(color='green')))
        fig.add_trace(go.Scatter(y=st.session_state['history_dis'], mode='lines', name='Dissonance (Pain)', line=dict(color='red')))
        fig.update_layout(title="Emotional State", xaxis_title="Step", yaxis_title="Intensity", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Quantum Plot
    if 'history_ent' in st.session_state:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=st.session_state['history_ent'], mode='lines', name='Confidence', line=dict(color='blue')))
        fig2.update_layout(title="Quantum Collapse (Confidence)", xaxis_title="Step", yaxis_title="Nats", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Sleep Mode Control
    st.divider()
    st.subheader("Sleep Cycle Control")
    st.markdown("Consolidate memories from Dream Core.")

    if st.button("Enter Sleep Mode (Dream)"):
        with st.status("Sleeping...") as status:
            st.write("Initializing Passive Pipeline...")
            msg = model.enter_idle_mode(interval=0.1)
            st.write(msg)
            time.sleep(1)
            st.write("Generating Dreams...")
            time.sleep(2)
            st.write("Consolidating Topological Memory...")
            time.sleep(1)
            msg_wake = model.exit_idle_mode()
            st.write(msg_wake)
            status.update(label="Sleep Cycle Complete", state="complete")
        st.success("Memories Consolidated.")

# Footer
st.markdown("---")
st.caption("Project MUSE | Phase 4 | O(N) Physics-Based AI")
