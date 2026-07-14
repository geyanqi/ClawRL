"""Public-behavior tests for the initial ClawRL framework."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from clawrl.system import AutomationDomain, default_manifest


class SystemManifestTests(unittest.TestCase):
    def test_default_manifest_covers_the_rl_lifecycle(self) -> None:
        manifest = default_manifest()

        self.assertEqual(manifest.name, "ClawRL")
        self.assertEqual(set(manifest.domains), set(AutomationDomain))
        self.assertEqual(len(manifest.domains), 7)

    def test_module_cli_emits_json_manifest(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "src")

        completed = subprocess.run(
            [sys.executable, "-m", "clawrl", "--json"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["name"], "ClawRL")
        self.assertIn("agentic-reward-generation", payload["domains"])
        self.assertIn("model-evaluation", payload["domains"])


if __name__ == "__main__":
    unittest.main()
