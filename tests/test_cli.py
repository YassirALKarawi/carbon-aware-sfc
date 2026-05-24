"""Tests for CLI entry point."""
import subprocess

def test_help():
    r = subprocess.run(["python", "lcavo_sim.py", "--help"],
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert "seeds" in r.stdout.lower() or "usage" in r.stdout.lower()
