"""Tests for CLI entry point."""
import subprocess

def test_help():
    r = subprocess.run(["python", "lcavo_sim.py", "--help"],
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert "seeds" in r.stdout.lower() or "usage" in r.stdout.lower()

def test_help_lists_qlcavo():
    """--help must advertise QL-CAVO (the paper's method name, not DRL-CAVO)."""
    r = subprocess.run(["python", "lcavo_sim.py", "--help"],
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert "QL-CAVO" in r.stdout
    assert "DRL-CAVO" not in r.stdout

def test_cli_rejects_legacy_drlcavo():
    """Legacy DRL-CAVO method name should be rejected after the QL-CAVO rename."""
    r = subprocess.run(
        ["python", "lcavo_sim.py", "--methods", "DRL-CAVO", "--quick",
         "--topology", "NSFNET", "--load", "Low", "--skip-figures"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode != 0
    assert "Unknown method" in (r.stdout + r.stderr) or "DRL-CAVO" in (r.stdout + r.stderr)
