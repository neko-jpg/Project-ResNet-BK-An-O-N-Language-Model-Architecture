"""
Local Boundary Core: The Reality Anchor

This module implements the LocalFileBoundary, which grounds the AI's imagination
in the reality of the repository's files (package.json, requirements.txt, etc.).

It enforces the "Strict Pull Strategy" by ensuring the AI is aware of
installed libraries and project constraints.
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path

class LocalFileBoundary:
    """
    Real Boundary Core.
    Reads local configuration files to define the "Universe" of the project.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.knowledge_base: List[str] = []
        self.allowed_libraries: List[str] = []
        self._load_reality()

    def _load_reality(self):
        """
        Load the 'Physical Constants' of the project (config files).
        """
        self.knowledge_base.append(f"Project Root: {self.root_dir.absolute()}")

        # 1. Load package.json (Node.js)
        pkg_path = self.root_dir / "package.json"
        if pkg_path.exists():
            try:
                with open(pkg_path, 'r') as f:
                    data = json.load(f)
                    deps = data.get('dependencies', {})
                    dev_deps = data.get('devDependencies', {})
                    self.allowed_libraries.extend(deps.keys())
                    self.allowed_libraries.extend(dev_deps.keys())
                    self.knowledge_base.append(f"Node.js Dependencies: {', '.join(deps.keys())}")
            except Exception as e:
                self.knowledge_base.append(f"Error reading package.json: {e}")

        # 2. Load requirements.txt (Python)
        req_path = self.root_dir / "requirements.txt"
        if req_path.exists():
            try:
                with open(req_path, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
                    # Simple parsing of package names (ignoring versions)
                    libs = [l.split('==')[0].split('>=')[0].split('<')[0] for l in lines]
                    self.allowed_libraries.extend(libs)
                    self.knowledge_base.append(f"Python Dependencies: {', '.join(libs)}")
            except Exception as e:
                self.knowledge_base.append(f"Error reading requirements.txt: {e}")

        # 3. Load README.md (General Rules)
        readme_path = self.root_dir / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, 'r') as f:
                    content = f.read()
                    # Store only the first 2000 chars to avoid context overflow in this simple implementation
                    self.knowledge_base.append(f"README Context: {content[:2000]}...")
            except Exception as e:
                self.knowledge_base.append(f"Error reading README.md: {e}")

    def fetch(self, query: str = "", k: int = 3) -> List[str]:
        """
        Fetch relevant constraints from the local file system.
        """
        # If query asks for imports or libraries, prioritize dependency lists
        if "import" in query.lower() or "library" in query.lower() or "install" in query.lower() or "use" in query.lower():
            # Return library constraints
            return [f"ALLOWED LIBRARIES: {', '.join(self.allowed_libraries)}"]

        # Otherwise return general knowledge (Simple search)
        relevant = []
        query_lower = query.lower()
        for doc in self.knowledge_base:
            if any(word in doc.lower() for word in query_lower.split()):
                relevant.append(doc)

        if not relevant:
            # If nothing specific found, return the most important context (Dependencies)
            return [f"Project Context: {len(self.allowed_libraries)} libraries installed."]

        return relevant[:k]

    def is_allowed_library(self, lib_name: str) -> bool:
        """
        Check if a library is allowed (exists in config).
        """
        # Standard libs (simplified list)
        std_libs = ['os', 'sys', 'json', 'math', 'random', 'time', 'datetime', 'pathlib', 'typing', 're']
        if lib_name in std_libs:
            return True

        return lib_name in self.allowed_libraries
