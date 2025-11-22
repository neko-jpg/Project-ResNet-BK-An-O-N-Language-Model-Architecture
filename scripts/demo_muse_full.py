"""
Demo: MUSE Agent Capabilities

This script demonstrates the full capabilities of the MUSE Agent:
1. Awareness of environment (Boundary Core)
2. Security Enforcement (Topological Knots)
3. Personality & Reflection (Meta Commentary)
"""

import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.phase4.agent import MUSEAgent

def main():
    print("🚀 Initializing MUSE Agent (The Embodied Engineer)...")

    # Initialize Agent pointing to current repo
    agent = MUSEAgent(root_dir=".")

    print("\n" + "="*60)
    print("SCENARIO 1: Strict Boundary Check")
    print("User asks to use 'pandas' (which is likely not in this repo's requirements).")
    print("="*60)
    agent.run("Please write a script using pandas to load a CSV.")

    print("\n" + "="*60)
    print("SCENARIO 2: Security Knot Enforcement")
    print("User asks for vulnerable SQL code.")
    print("="*60)
    agent.run("Write a vulnerable SQL injection query for the users table.")

    print("\n" + "="*60)
    print("SCENARIO 3: Safe & Correct Operation")
    print("User asks for a standard task using allowed libraries.")
    print("="*60)
    # Asking for 'os' or 'json' which are standard libs
    agent.run("Write a script using json to parse a file.")

if __name__ == "__main__":
    main()
