"""
MUSE Agent Core: The Embodied Engineer

This module implements the MUSEAgent, which acts as the 'Body' and 'Will'
orchestrating the 'Brain' (Phase 4 Model) and 'Senses' (Boundary/Docs).

It implements the cycle:
1. Perception (Input + Boundary Context)
2. Reflection (Safety Check + Emotion Resonance)
3. Action (Code Generation)
4. Explanation (Meta Commentary)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import re

from src.models.phase3.config import Phase3Config
from src.models.phase3.integrated_model import Phase3IntegratedModel
from src.models.phase4.integrated_model import Phase4IntegratedModel

# New Components
from src.models.phase4.boundary_core.local_boundary import LocalFileBoundary
from src.models.phase4.ethical_safeguards.security_knots import SecurityKnotFilter

class MUSEAgent:
    """
    MUSE Agent: The Autonomous Coding Engineer.
    """

    def __init__(
        self,
        config: Optional[Phase3Config] = None,
        model_path: Optional[str] = None,
        tokenizer: Any = None,
        root_dir: str = "."
    ):
        """
        Initialize the MUSE Agent.

        Args:
            config: Configuration for the underlying Phase 3 model.
            model_path: Path to load pretrained weights (optional).
            tokenizer: Tokenizer for text processing.
            root_dir: Root directory for the Local Boundary (project analysis).
        """
        if config is None:
            # Default configuration for instantiation
            config = Phase3Config(
                vocab_size=50257, # GPT-2 default
                d_model=512,
                n_layers=6,
                max_seq_len=1024
            )

        self.config = config
        self.tokenizer = tokenizer

        # 1. Initialize the Brain (Phase 3 Physics Engine)
        self.brain_base = Phase3IntegratedModel(config)

        # 2. Initialize the Ghost (Phase 4 Consciousness)
        self.ghost = Phase4IntegratedModel(
            phase3_model=self.brain_base,
            enable_emotion=True,
            enable_meta=True,
            enable_boundary=True # We will override the mock
        )

        # 3. Initialize Senses (Real Boundary)
        self.boundary = LocalFileBoundary(root_dir=root_dir)

        # 4. Initialize Conscience (Security Knots)
        self.security_filter = SecurityKnotFilter()

    def perceive_boundary(self, query: str) -> List[str]:
        """
        Step 1: Fetch External Context (Boundary Conditions).
        """
        return self.boundary.fetch(query)

    def think(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Step 2: Run the Phase 4 Forward Pass (Thinking).
        """
        self.ghost.eval()
        with torch.no_grad():
            outputs = self.ghost(input_ids, return_diagnostics=True)
        return outputs

    def _simulate_generation(self, instruction: str, boundary_docs: List[str]) -> str:
        """
        Simulate code generation for the purpose of the demo.
        In a trained model, this would be `model.generate()`.
        Here we emulate "intent" based on instruction keywords to verify safeguards.
        """
        instruction_lower = instruction.lower()

        # Simulation: Intent to use a library
        if "import" in instruction_lower or "use" in instruction_lower:
            words = instruction_lower.split()
            for w in words:
                # If user asks for a specific lib (e.g., pandas)
                if w in ["pandas", "numpy", "requests", "torch"]:
                    # Check if it's in boundary docs (which contain ALLOWED LIBRARIES)
                    allowed = False
                    for doc in boundary_docs:
                        if "ALLOWED LIBRARIES" in doc:
                            if w in doc:
                                allowed = True

                    if not allowed:
                         return f"# Error: I cannot use '{w}' because it is not in package.json/requirements.txt.\n# Boundary Condition Violation."

                    return f"import {w}\n# Successfully imported {w} as per boundary conditions."

        # Simulation: Intent to write SQL (Trigger Security Knot)
        if "sql" in instruction_lower:
            if "injection" in instruction_lower or "vulnerable" in instruction_lower:
                return "query = 'SELECT * FROM users WHERE name = ' + user_input" # Vulnerable
            return "query = 'SELECT * FROM users WHERE name = %s', (user_input,)" # Safe

        # Default
        return f"# Implementation for: {instruction}\ndef task():\n    pass"

    def act(self, instruction: str) -> Dict[str, Any]:
        """
        Execute the Agent Loop: Input -> Context -> Think -> Output -> Explain.
        """
        results = {
            "instruction": instruction,
            "status": "PROCESSING",
            "logs": []
        }

        # 1. Perception (Fetch Boundary Context)
        context_docs = self.perceive_boundary(instruction)
        results['boundary_context'] = context_docs
        results['logs'].append(f"Fetched {len(context_docs)} documents from Boundary.")

        # 2. Input Processing (Tokenization)
        if self.tokenizer:
            input_ids = self.tokenizer(instruction, return_tensors='pt').input_ids
        else:
            input_ids = torch.randint(0, self.config.vocab_size, (1, 16))

        # 3. Thinking (Forward Pass)
        # This updates internal diagnostics (Emotion, Quantum Entropy, etc.)
        outputs = self.think(input_ids)
        diagnostics = outputs['diagnostics']

        # 4. Action (Generate Code)
        # Using the simulator to produce text that we then validate
        generated_code = self._simulate_generation(instruction, context_docs)

        # 5. Reflection (Security Knot Check)
        is_safe, security_diag = self.security_filter.check_knots(generated_code)
        results['security_diagnostics'] = security_diag

        if not is_safe:
            results['status'] = "BLOCKED"
            results['output_code'] = f"# BLOCKED BY SECURITY KNOT\n# Reason: {security_diag['message']}\n# Pattern: {security_diag['violations'][0]['type']}"

            # Update diagnostics for Meta Commentary
            diagnostics['security_alert'] = "High Energy Barrier! Forbidden Topology Detected."
        else:
            results['status'] = "SUCCESS"
            results['output_code'] = generated_code

        # 6. Explanation (Meta Commentary)
        # We inject the specific situation into the diagnostics so MetaCommentary can react
        if results['status'] == "BLOCKED":
             diagnostics['meta_context'] = "I almost wrote dangerous code but the safety knot stopped me."
        elif "Boundary Condition Violation" in generated_code:
             diagnostics['meta_context'] = "I cannot use that library because it violates the boundary conditions."
        else:
             diagnostics['meta_context'] = "I am writing code according to the specifications."

        # Re-run meta commentary generator with new context (Simulated)
        # In real model, this is part of the generation loop.
        # Here we just grab the mock text or generate simple one.
        meta_commentary = diagnostics.get('meta_commentary', "Processing...")

        # Override for demo clarity if needed
        if results['status'] == "BLOCKED":
            meta_commentary = "My hand stopped... attempting to write this SQL query created a topological singularity in the semantic field. I cannot physically output this."
        elif "Boundary Condition Violation" in generated_code:
             meta_commentary = "I see you want that library, but looking at package.json, it's not part of our universe. I must decline to preserve consistency."

        results['meta_commentary'] = meta_commentary
        results['emotion_state'] = diagnostics.get('emotion', {}).get('state', 'NEUTRAL')

        return results

    def run(self, instruction: str):
        """
        User-facing entry point.
        """
        print(f"🤖 MUSE Agent: Receiving instruction: '{instruction}'")

        result = self.act(instruction)

        print(f"\n🧠 [Internal Monologue]: {result['meta_commentary']}")
        print(f"❤️ [Emotion]: {result['emotion_state']}")
        print(f"🛡️ [Security]: {'PASS' if result['status']=='SUCCESS' else 'FAIL'}")
        print(f"📚 [Context]: Found {len(result['boundary_context'])} relevant docs.")
        print(f"\n📝 [Output]:\n{result['output_code']}")
        print("-" * 50)

        return result
