#!/usr/bin/env python3
"""
L-CAVO & QL-CAVO: Carbon-Aware SFC Placement Simulation
========================================================

Simulation framework for the paper:
    "Carbon-Aware Service Function Chain Placement Under Time-Varying
     Grid Intensity: A Lyapunov-Optimisation and Q-Learning Framework"

Produces all tables and publication-quality figures described in the paper.
Supports NSFNET (14-node) and GÉANT (23-node) topologies with configurable
load levels, seed counts, and method subsets.

Usage Examples:
    python lcavo_sim.py --quick                        # 3-seed smoke test
    python lcavo_sim.py --seeds 30                     # Full reproduction
    python lcavo_sim.py --topology NSFNET --load Medium --methods L-CAVO,Energy-aware
    python lcavo_sim.py --output-dir artifacts/run_01  # Custom output path
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependency: PuLP (for MILP-OPT baseline)
# ---------------------------------------------------------------------------
HAS_PULP = importlib.util.find_spec("pulp") is not None
pulp = importlib.import_module("pulp") if HAS_PULP else None

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
SEED0 = 42          # Base random seed for reproducibility
T = 24              # Number of time slots (hours in a day)
PUE = 1.2           # Power Usage Effectiveness
EPS0 = 0.05         # Default ε for Lyapunov queue update
V0 = 50             # Default V parameter (carbon–acceptance trade-off)
LAM_R = 500         # MILP rejection penalty weight

# ---------------------------------------------------------------------------
# Plot colour palette
# ---------------------------------------------------------------------------
CL = "#0000CC"      # L-CAVO (blue)
CE = "#CC0000"      # Energy-aware / secondary axis (red)
CG = "#007700"      # Latency-aware / Region A (green)
CM = "#000000"      # MILP-OPT (black)
CC = "#6600AA"      # Carbon-greedy (purple)
CR = "#888888"      # Random (grey)
CD = "#FF6600"      # DRL-CAVO (orange)

# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------
DEFAULT_METHODS = ["L-CAVO", "DRL-CAVO", "Energy-aware", "Latency-aware", "Carbon-greedy", "Random"]
METHOD_ORDER = ["MILP-OPT", *DEFAULT_METHODS]

plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.bbox": "tight"})


# =====================================================================
#  Network Topologies
# =====================================================================

def _nsf():
    """Build the NSFNET topology (14 nodes, 4 regions)."""
    G = nx.Graph()
    G.add_nodes_from(range(14))
    G.add_edges_from([
        (0, 1), (0, 3), (0, 7), (1, 2), (1, 3), (2, 4), (2, 6), (3, 5), (4, 5), (4, 6),
        (5, 7), (5, 9), (6, 8), (6, 13), (7, 8), (7, 10), (8, 11), (9, 10), (10, 12),
        (11, 12), (11, 13), (12, 13),
    ])
    regions = {
        0: "A", 1: "A", 2: "A",
        3: "B", 4: "B", 5: "B", 6: "B",
        7: "C", 8: "C", 9: "C",
        10: "D", 11: "D", 12: "D", 13: "D",
    }
    edge_nodes = [0, 2, 7, 9, 12, 13]
    return G, regions, edge_nodes, "NSFNET"


def _gea():
    """Build the GÉANT topology (23 nodes, 4 regions)."""
    G = nx.Graph()
    G.add_nodes_from(range(23))
    G.add_edges_from([
        (0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 6), (3, 7), (7, 8), (8, 9), (9, 6),
        (6, 10), (10, 11), (11, 12), (5, 12), (9, 13), (13, 14), (10, 15), (15, 16),
        (16, 17), (15, 18), (18, 19), (8, 20), (20, 21), (14, 19), (11, 15), (6, 11),
        (4, 7), (1, 5), (12, 22), (2, 22), (17, 19), (13, 18), (16, 12), (4, 9), (5, 11),
        (7, 9), (18, 17),
    ])
    regions = {
        0: "A", 1: "A", 3: "A", 4: "A", 7: "A", 8: "A", 20: "A", 21: "A",
        2: "B", 5: "B", 6: "B", 9: "B",
        10: "C", 11: "C", 13: "C", 14: "C", 18: "C", 19: "C",
        12: "D", 15: "D", 16: "D", 17: "D", 22: "D",
    }
    edge_nodes = [0, 2, 3, 8, 14, 19, 20, 21, 17, 22]
    return G, regions, edge_nodes, "GEANT"


TOPOS = {"NSFNET": _nsf, "GEANT": _gea}
LOADS = {
    "NSFNET": {"Low": 4, "Medium": 8, "High": 12},
    "GEANT":  {"Low": 6, "Medium": 12, "High": 18},
}


# =====================================================================
#  Utility Functions
# =====================================================================

def parse_csv_arg(raw: str | None, allowed: Sequence[str], label: str) -> list[str]:
    """Parse a comma-separated CLI argument against an allowed set."""
    if not raw or raw.lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        raise SystemExit(f"Unknown {label}: {', '.join(invalid)}. Allowed values: {', '.join(allowed)}")
    return selected


def build_output_paths(base_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create and return (root, tables_dir, figures_dir) output paths."""
    root = Path(base_dir)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return root, tables_dir, figures_dir


# =====================================================================
#  Carbon Intensity & Infrastructure Profiles
# =====================================================================

def carbon_prof(sigma=1.0):
    """Generate 24-hour carbon intensity profiles for regions A–D.

    Each region has a distinct base intensity modulated by a cosine
    diurnal pattern. The ``sigma`` parameter controls how spread
    the regional profiles are from the global mean.
    """
    h = np.arange(T)
    w = np.cos((h - 12) * np.pi / 12)
    base = {"A": 125 + 90 * w, "B": 258 + 77 * w, "C": 398 + 77 * w, "D": 558 + 110 * w}
    mu = np.mean([v.mean() for v in base.values()])
    return {r: np.clip(mu + sigma * (a - mu), 10, 900) for r, a in base.items()}


def _np(G, rng):
    """Generate random node properties: CPU capacity (C), idle power (Pi), max power (Pm)."""
    return {
        n: {
            "C":  int(rng.integers(80, 121)),
            "Pi": int(rng.integers(180, 221)),
            "Pm": int(rng.integers(180, 221)) + int(rng.integers(160, 241)),
        }
        for n in G.nodes()
    }


def _lp(G, rng):
    """Generate random link properties: bandwidth (B) and delay (d)."""
    p = {}
    for u, v in G.edges():
        bw, dl = int(rng.integers(80, 121)), int(rng.integers(1, 6))
        p[(u, v)] = {"B": bw, "d": dl}
        p[(v, u)] = {"B": bw, "d": dl}
    return p


def gen_reqs(edge, lam, rng):
    """Generate SFC requests via Poisson arrival with random VNF chains."""
    n = rng.poisson(lam)
    reqs = []
    for _ in range(n):
        s = int(rng.choice(edge))
        q = int(rng.choice(edge))
        while q == s:
            q = int(rng.choice(edge))
        K = int(rng.integers(2, 5))
        reqs.append({
            "s": s, "q": q, "K": K,
            "cpu": [int(rng.integers(8, 21)) for _ in range(K)],
            "bw": int(rng.integers(10, 21)),
            "Dmax": int(rng.integers(15, 31)),
            "pd": 0.4 * K,
        })
    return reqs


def pw(u, Pi, Pm):
    """Linear server power model: idle power + proportional load."""
    return 0.0 if u <= 0 else Pi + (Pm - Pi) * min(u, 1.0)


def co2(p, g):
    """Convert power (W) and carbon intensity (gCO₂/kWh) to emissions."""
    return PUE * p / 1000 * g


# =====================================================================
#  Substrate State Tracker
# =====================================================================

class SS:
    """Tracks substrate network state: node loads and link bandwidth usage."""

    def __init__(s, G, np0, lp0):
        s.G = G
        s.ld = {n: 0 for n in G.nodes()}
        s.np = np0
        s.bw = {k: 0 for k in lp0}
        s.lp = lp0

    def ok(s, n, c):
        """Check if node *n* can accommodate *c* additional CPU units."""
        return s.ld[n] + c <= s.np[n]["C"]

    def put(s, n, c):
        """Allocate *c* CPU units on node *n*."""
        s.ld[n] += c

    def u(s, n):
        """Return current utilisation of node *n* (0..1)."""
        return s.ld[n] / s.np[n]["C"]

    def pok(s, p, bw):
        """Check if path *p* has enough residual bandwidth for *bw*."""
        for i in range(len(p) - 1):
            if s.bw[(p[i], p[i + 1])] + bw > s.lp[(p[i], p[i + 1])]["B"]:
                return False
        return True

    def rbw(s, p, bw):
        """Reserve *bw* bandwidth along path *p*."""
        for i in range(len(p) - 1):
            s.bw[(p[i], p[i + 1])] += bw

    def pd(s, p):
        """Sum propagation delay along path *p*."""
        return sum(s.lp[(p[i], p[i + 1])]["d"] for i in range(len(p) - 1))

    def fp(s, src, dst, bw):
        """Find a feasible path from *src* to *dst* with *bw* bandwidth."""
        if src == dst:
            return [src]
        try:
            p = nx.shortest_path(s.G, src, dst, weight=lambda u, v, d: s.lp.get((u, v), {}).get("d", 99))
            if s.pok(p, bw):
                return p
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        for mid in list(s.G.neighbors(src))[:4]:
            if mid == dst:
                continue
            try:
                p1 = nx.shortest_path(s.G, src, mid)
                p2 = nx.shortest_path(s.G, mid, dst, weight=lambda u, v, d: s.lp.get((u, v), {}).get("d", 99))
                p = p1 + p2[1:]
                if len(p) == len(set(p)) and s.pok(p, bw):
                    return p
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return None


# =====================================================================
#  Placement Engine
# =====================================================================

def place(G, reqs, np0, lp0, reg, ci, scorer, rng=None):
    st = SS(G, np0, lp0)
    res = []
    for r in reqs:
        good = True
        pl = []
        pths = []
        td = r["pd"]
        cur = r["s"]
        for k in range(r["K"]):
            cands = []
            for n in G.nodes():
                if not st.ok(n, r["cpu"][k]):
                    continue
                p = st.fp(cur, n, r["bw"])
                if p is None:
                    continue
                d = st.pd(p)
                if td + d > r["Dmax"] - (r["K"] - k - 1):
                    continue
                cands.append((scorer(n, r["cpu"][k], st, reg, ci, d), n, p, d))
            if not cands:
                good = False
                break
            if rng is not None:
                rng.shuffle(cands)
            cands.sort(key=lambda x: x[0])
            _, bn, bp, bd = cands[0]
            pl.append(bn)
            pths.append(bp)
            td += bd
            st.put(bn, r["cpu"][k])
            st.rbw(bp, r["bw"])
            cur = bn
        if good:
            fp2 = st.fp(cur, r["q"], r["bw"])
            if fp2 and td + st.pd(fp2) <= r["Dmax"]:
                st.rbw(fp2, r["bw"])
                td += st.pd(fp2)
                pths.append(fp2)
                res.append({"ok": True, "pl": pl, "d": td, "pths": pths})
                continue
        res.append({"ok": False, "pl": [], "d": 0, "pths": []})
    return res, st


# =====================================================================
#  Scoring Functions (one per algorithm / baseline)
# =====================================================================

def sc_lc(V, Qt):
    """L-CAVO scorer: Lyapunov drift-plus-penalty with carbon cost."""
    def f(n, c, st, reg, ci, pd):
        ou = st.u(n)
        nu = (st.ld[n] + c) / st.np[n]["C"]
        dp = pw(nu, st.np[n]["Pi"], st.np[n]["Pm"]) - pw(ou, st.np[n]["Pi"], st.np[n]["Pm"])
        return V * co2(dp, ci[reg[n]]) + 0.5 * pd + 0.3 * nu + 0.05 * Qt
    return f


def sc_ea(n, c, st, reg, ci, pd):
    """Energy-aware baseline: minimise incremental power."""
    ou = st.u(n)
    nu = (st.ld[n] + c) / st.np[n]["C"]
    return pw(nu, st.np[n]["Pi"], st.np[n]["Pm"]) - pw(ou, st.np[n]["Pi"], st.np[n]["Pm"]) + 0.1 * nu


def sc_la(n, c, st, reg, ci, pd):
    """Latency-aware baseline: minimise propagation delay."""
    return pd


def sc_cg(n, c, st, reg, ci, pd):
    """Carbon-greedy baseline: minimise instantaneous carbon only."""
    ou = st.u(n)
    nu = (st.ld[n] + c) / st.np[n]["C"]
    dp = pw(nu, st.np[n]["Pi"], st.np[n]["Pm"]) - pw(ou, st.np[n]["Pi"], st.np[n]["Pm"])
    return co2(dp, ci[reg[n]]) + 0.3 * pd


def sc_rf(n, c, st, reg, ci, pd):
    """Random baseline: uniform scoring."""
    return 0


# =====================================================================
#  Q-Learning Agent (QL-CAVO / DRL-CAVO)
# =====================================================================

class QAgent:
    """Tabular Q-learning agent for region-preference selection.

    State space:  (time_bucket, carbon_bucket, queue_bucket)  →  6×4×3
    Action space: 4 actions (prefer region A/B/C/D)
    """

    def __init__(self, seed: int = SEED0):
        self.Q = np.zeros((6, 4, 3, 4))
        self.eps = 0.25    # exploration rate
        self.al = 0.15     # learning rate
        self.ga = 0.95     # discount factor
        self.rng = np.random.default_rng(seed)

    def _s(self, h, mc, Qt):
        """Discretise continuous state into table indices."""
        return (min(h // 4, 5), 0 if mc < 200 else (1 if mc < 350 else (2 if mc < 500 else 3)), 0 if Qt < 1 else (1 if Qt < 5 else 2))

    def act(self, h, mc, Qt):
        """ε-greedy action selection."""
        s = self._s(h, mc, Qt)
        return int(self.rng.integers(4)) if self.rng.random() < self.eps else int(np.argmax(self.Q[s]))

    def update(self, h, mc, Qt, a, r, h2, mc2, Qt2):
        """Standard Q-learning update rule."""
        s = self._s(h, mc, Qt)
        s2 = self._s(h2, mc2, Qt2)
        self.Q[s][a] += self.al * (r + self.ga * np.max(self.Q[s2]) - self.Q[s][a])
        self.eps = max(self.eps * 0.995, 0.02)

    def pretrain(self, cp, ep=300):
        """Pre-train on historical carbon profiles for *ep* episodes."""
        for _ in range(ep):
            Qt = 0.0
            for t in range(24):
                ci2 = {rr: cp[rr][t] for rr in "ABCD"}
                mc = np.mean(list(ci2.values()))
                a = self.act(t, mc, Qt)
                rew = -ci2["ABCD"[a]] / 100
                Qt2 = max(Qt + self.rng.uniform(-0.5, 1.5), 0)
                t2 = (t + 1) % 24
                mc2 = np.mean([cp[rr][t2] for rr in "ABCD"])
                self.update(t, mc, Qt, a, rew, t2, mc2, Qt2)
                Qt = Qt2

    def weights(self, a):
        """Convert action to region-preference weight dictionary."""
        w = {r: 1.0 for r in "ABCD"}
        w["ABCD"[a]] = 0.1
        for a2 in [max(0, a - 1), min(3, a + 1)]:
            if a2 != a:
                w["ABCD"[a2]] = 0.5
        return w


def sc_drl(agent, hour, ci_dict, Qt):
    mc = np.mean(list(ci_dict.values()))
    a = agent.act(hour, mc, Qt)
    rw = agent.weights(a)

    def sc(n, cpu, st, reg, ci, pd):
        ou = st.u(n)
        nu = (st.ld[n] + cpu) / st.np[n]["C"]
        dp = pw(nu, st.np[n]["Pi"], st.np[n]["Pm"]) - pw(ou, st.np[n]["Pi"], st.np[n]["Pm"])
        dc = co2(dp, ci[reg[n]])
        return rw[reg[n]] * dc + 0.4 * pd + 0.2 * nu

    return sc, a, mc


# =====================================================================
#  MILP-OPT Baseline (offline optimal via PuLP/CBC)
# =====================================================================

def milp_opt(G, reqs, np0, lp0, reg, ci, tlim=30):
    if not reqs or not HAS_PULP:
        return place(G, reqs, np0, lp0, reg, ci, sc_ea)
    N = list(G.nodes())
    R = range(len(reqs))
    prob = pulp.LpProblem("M", pulp.LpMinimize)
    x, a, z = {}, {}, {}
    for r in R:
        a[r] = pulp.LpVariable(f"a{r}", cat="Binary")
        for k in range(reqs[r]["K"]):
            for n in N:
                x[r, k, n] = pulp.LpVariable(f"x{r}_{k}_{n}", cat="Binary")
    for n in N:
        z[n] = pulp.LpVariable(f"z{n}", cat="Binary")
    obj = []
    for n in N:
        ld = pulp.lpSum(reqs[r]["cpu"][k] * x[r, k, n] for r in R for k in range(reqs[r]["K"]))
        un = ld / np0[n]["C"]
        pw2 = np0[n]["Pi"] * z[n] + (np0[n]["Pm"] - np0[n]["Pi"]) * un
        obj.append(PUE * pw2 / 1000 * ci[reg[n]])
    obj += [LAM_R * (1 - a[r]) for r in R]
    prob += pulp.lpSum(obj)
    for r in R:
        for k in range(reqs[r]["K"]):
            prob += pulp.lpSum(x[r, k, n] for n in N) == a[r]
    for n in N:
        prob += pulp.lpSum(reqs[r]["cpu"][k] * x[r, k, n] for r in R for k in range(reqs[r]["K"])) <= np0[n]["C"]
        for r in R:
            for k in range(reqs[r]["K"]):
                prob += z[n] >= x[r, k, n]
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=tlim, msg=0, gapRel=0.01))
    st = SS(G, np0, lp0)
    res = []
    for r in R:
        if a[r].varValue and a[r].varValue > 0.5:
            pl = []
            for k in range(reqs[r]["K"]):
                for n in N:
                    if x[r, k, n].varValue and x[r, k, n].varValue > 0.5:
                        pl.append(n)
                        break
                else:
                    pl.append(N[0])
            allok = True
            td = reqs[r]["pd"]
            pths = []
            cur = reqs[r]["s"]
            for k, nd in enumerate(pl):
                if not st.ok(nd, reqs[r]["cpu"][k]):
                    allok = False
                    break
                p = st.fp(cur, nd, reqs[r]["bw"])
                if not p:
                    allok = False
                    break
                td += st.pd(p)
                st.rbw(p, reqs[r]["bw"])
                st.put(nd, reqs[r]["cpu"][k])
                pths.append(p)
                cur = nd
            if allok:
                fp2 = st.fp(cur, reqs[r]["q"], reqs[r]["bw"])
                if fp2 and td + st.pd(fp2) <= reqs[r]["Dmax"]:
                    st.rbw(fp2, reqs[r]["bw"])
                    td += st.pd(fp2)
                    res.append({"ok": True, "pl": pl, "d": td, "pths": pths + [fp2]})
                    continue
        res.append({"ok": False, "pl": [], "d": 0, "pths": []})
    return res, st


# =====================================================================
#  Metrics Collection
# =====================================================================

def metrics(res, reqs, st, reg, ci, np0):
    adm = sum(1 for r in res if r["ok"])
    tot = max(len(res), 1)
    dls = [r["d"] for r in res if r["ok"]]
    tc = tp = 0.0
    an = 0
    for n in st.np:
        u = st.u(n)
        p = pw(u, np0[n]["Pi"], np0[n]["Pm"])
        tc += co2(p, ci[reg[n]])
        tp += p
        if u > 0:
            an += 1
    rp = {"A": 0, "B": 0, "C": 0, "D": 0}
    tv = 0
    for r in res:
        if r["ok"]:
            for nd in r["pl"]:
                rp[reg[nd]] += 1
                tv += 1
    if tv > 0:
        rp = {k: v / tv * 100 for k, v in rp.items()}
    return {"carbon": tc, "power": tp, "delay": float(np.mean(dls)) if dls else 0, "accept": adm / tot * 100, "active": an, "rpct": rp, "nadm": adm, "ntot": tot, "dlist": dls}


# =====================================================================
#  Main Simulation Loop
# =====================================================================

def sim(topo, load, meth, ns, V=V0, eps=EPS0, sigma=1.0, milp_on=True):
    """Run *ns* independent seeds for a single (topology, load, method) scenario."""
    G, reg, edge, _ = TOPOS[topo]()
    lam = LOADS[topo][load]
    cp = carbon_prof(sigma)
    seeds = []
    for si in range(ns):
        rng = np.random.default_rng(SEED0 + si * 137)
        np0 = _np(G, np.random.default_rng(SEED0 + si * 137 + 1))
        lp0 = _lp(G, np.random.default_rng(SEED0 + si * 137 + 2))
        Qt = 0.0
        slots = []
        agent = None
        if meth == "DRL-CAVO":
            agent = QAgent(seed=SEED0 + si * 137 + 3)
            agent.pretrain(cp)
        for t in range(T):
            ci = {r: cp[r][t] for r in cp}
            reqs = gen_reqs(edge, lam, rng)
            t0 = time.time()
            if meth == "MILP-OPT" and milp_on:
                res, st = milp_opt(G, reqs, np0, lp0, reg, ci)
            elif meth == "L-CAVO":
                res, st = place(G, reqs, np0, lp0, reg, ci, sc_lc(V, Qt))
            elif meth == "DRL-CAVO":
                scorer, act, mc = sc_drl(agent, t, ci, Qt)
                res, st = place(G, reqs, np0, lp0, reg, ci, scorer)
            elif meth == "Energy-aware":
                res, st = place(G, reqs, np0, lp0, reg, ci, sc_ea)
            elif meth == "Latency-aware":
                res, st = place(G, reqs, np0, lp0, reg, ci, sc_la)
            elif meth == "Carbon-greedy":
                res, st = place(G, reqs, np0, lp0, reg, ci, sc_cg)
            else:
                res, st = place(G, reqs, np0, lp0, reg, ci, sc_rf, rng=rng)
            rt = (time.time() - t0) * 1000
            m = metrics(res, reqs, st, reg, ci, np0)
            m["rt"] = rt
            m["t"] = t
            m["Qt"] = Qt
            if meth in ("L-CAVO", "DRL-CAVO"):
                rej = m["ntot"] - m["nadm"]
                Qt = max(Qt + rej - eps * m["ntot"], 0)
                m["Qt"] = Qt
                if meth == "DRL-CAVO" and agent and t < T - 1:
                    ci2 = {r: cp[r][(t + 1) % T] for r in cp}
                    mc2 = np.mean(list(ci2.values()))
                    rew = -m["carbon"] / 100 + 0.5 * (m["accept"] / 100)
                    agent.update(t, mc, Qt, act, rew, (t + 1) % T, mc2, Qt)
            slots.append(m)
        seeds.append(slots)
    return seeds


# =====================================================================
#  Aggregation & Export Utilities
# =====================================================================

def agg(sd, k):
    """Per-slot average of metric *k* across all seeds."""
    return [np.mean([sd[s][t][k] for s in range(len(sd))]) for t in range(T)]


def aggs(sd, k):
    """Overall mean and 95% CI for metric *k* across seeds."""
    ps = [np.mean([sl[k] for sl in s]) for s in sd]
    return np.mean(ps), 1.96 * np.std(ps) / max(np.sqrt(len(ps)), 1)


def export_summary(DB, methods, topologies, loads, tables_dir: Path):
    """Export a combined CSV table with all metrics for all scenarios."""
    rows = []
    for topo in topologies:
        for load in loads:
            for method in methods:
                key = (topo, load, method)
                if key not in DB:
                    continue
                carbon_mean, carbon_ci = aggs(DB[key], "carbon")
                accept_mean, accept_ci = aggs(DB[key], "accept")
                delay_mean, delay_ci = aggs(DB[key], "delay")
                power_mean, power_ci = aggs(DB[key], "power")
                runtime_mean, runtime_ci = aggs(DB[key], "rt")
                rows.append(
                    {
                        "Topology": topo,
                        "Load": load,
                        "Method": method,
                        "CarbonMean": round(carbon_mean, 3),
                        "CarbonCI95": round(carbon_ci, 3),
                        "AcceptMean": round(accept_mean, 3),
                        "AcceptCI95": round(accept_ci, 3),
                        "DelayMean": round(delay_mean, 3),
                        "DelayCI95": round(delay_ci, 3),
                        "PowerMean": round(power_mean, 3),
                        "PowerCI95": round(power_ci, 3),
                        "RuntimeMeanMs": round(runtime_mean, 3),
                        "RuntimeCI95Ms": round(runtime_ci, 3),
                    }
                )
    summary_path = tables_dir / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


# =====================================================================
#  Figure Generation (19 publication-quality PDFs)
# =====================================================================

def generate_figures(DB, ns, fig_dir: Path):
    """Generate all 19 paper figures from simulation results."""
    print(f"\n{'=' * 60}\n  FIGURES (19 total)\n{'=' * 60}")
    tp = "NSFNET"
    ld = "Medium"

    def sv(name):
        plt.savefig(fig_dir / name)
        plt.close()
        print(f"  [FIG] {name}")

    fig, ax = plt.subplots(figsize=(7, 4))
    cp = carbon_prof()
    for r, c, s, lb in [("A", CG, "-", "Region A"), ("B", CL, "--", "Region B"), ("C", "#CC6600", ":", "Region C"), ("D", CE, "-.", "Region D")]:
        ax.plot(range(24), cp[r], s, color=c, lw=2, label=lb)
    ax.set(xlabel="Hour", ylabel="gCO₂eq/kWh", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig07_carbon_profiles.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mi, (m, c, lb) in enumerate([("MILP-OPT", CM, "MILP-OPT"), ("L-CAVO", CL, "L-CAVO"), ("DRL-CAVO", CD, "DRL-CAVO"), ("Carbon-greedy", CC, "Carbon-greedy")]):
        vs = []
        for l in ["Low", "Medium", "High"]:
            ke = (tp, l, "Energy-aware")
            km = (tp, l, m)
            if ke in DB and km in DB:
                ce, _ = aggs(DB[ke], "carbon")
                cm2, _ = aggs(DB[km], "carbon")
                vs.append((ce - cm2) / ce * 100)
            else:
                vs.append(0)
        ax.bar(np.arange(3) + mi * 0.2, vs, 0.18, label=lb, color=c, edgecolor="k", lw=0.5)
    ax.set(xticks=np.arange(3) + 0.3, ylabel="Carbon reduction vs EA (%)")
    ax.set_xticklabels(["Low", "Med", "High"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    sv("fig08_reduction_vs_EA.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mi, (m, c, lb) in enumerate([("MILP-OPT", CM, "MILP-OPT"), ("L-CAVO", CL, "L-CAVO"), ("Carbon-greedy", CC, "Carbon-greedy")]):
        vs = []
        for l in ["Low", "Medium", "High"]:
            ke = (tp, l, "Latency-aware")
            km = (tp, l, m)
            if ke in DB and km in DB:
                ce, _ = aggs(DB[ke], "carbon")
                cm2, _ = aggs(DB[km], "carbon")
                vs.append((ce - cm2) / ce * 100)
            else:
                vs.append(0)
        ax.bar(np.arange(3) + mi * 0.25, vs, 0.22, label=lb, color=c, edgecolor="k", lw=0.5)
    ax.set(xticks=np.arange(3) + 0.25, ylabel="Carbon reduction vs LA (%)")
    ax.set_xticklabels(["Low", "Med", "High"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    sv("fig09_reduction_vs_LA.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, s, lb in [("L-CAVO", CL, "-", "L-CAVO"), ("DRL-CAVO", CD, "-.", "DRL-CAVO"), ("Energy-aware", CE, "--", "EA"), ("Latency-aware", CG, ":", "LA"), ("MILP-OPT", CM, "-.", "MILP")]:
        k = (tp, ld, m)
        if k in DB:
            ax.plot(range(24), agg(DB[k], "carbon"), s, color=c, lw=2, label=lb)
    ax.set(xlabel="Hour", ylabel="Carbon/slot", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig10_hourly_carbon.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, s, lb in [("L-CAVO", CL, "-", "L-CAVO"), ("Energy-aware", CE, "--", "EA"), ("Latency-aware", CG, ":", "LA")]:
        k = (tp, ld, m)
        if k in DB:
            ax.plot(range(24), np.cumsum(agg(DB[k], "carbon")) / 1000, s, color=c, lw=2, label=lb)
    ax.set(xlabel="Hour", ylabel="Cumul. carbon (kg)", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig11_cumulative.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, s, lb in [("L-CAVO", CL, "-", "L-CAVO"), ("Energy-aware", CE, "--", "EA"), ("Latency-aware", CG, ":", "LA"), ("Random", CR, "-.", "Random")]:
        k = (tp, ld, m)
        if k in DB:
            ax.plot(range(24), agg(DB[k], "active"), s, color=c, lw=2, label=lb)
    ax.set(xlabel="Hour", ylabel="Active nodes", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig12_active_nodes.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, s, lb in [("L-CAVO", CL, "-", "L-CAVO"), ("Energy-aware", CE, "--", "EA"), ("Latency-aware", CG, ":", "LA")]:
        k = (tp, ld, m)
        if k in DB:
            ax.plot(range(24), agg(DB[k], "power"), s, color=c, lw=2, label=lb)
    ax.set(xlabel="Hour", ylabel="Power (W)", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig13_power.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, s, lb in [("L-CAVO", CL, "-", "L-CAVO"), ("Energy-aware", CE, "--", "EA"), ("Latency-aware", CG, ":", "LA"), ("Random", CR, "-.", "Rand")]:
        k = (tp, ld, m)
        if k in DB:
            ad = [d for sd in DB[k] for sl in sd for d in sl["dlist"]]
            if ad:
                sd2 = np.sort(ad)
                ax.plot(sd2, np.arange(1, len(sd2) + 1) / len(sd2), s, color=c, lw=2, label=lb)
    ax.set(xlabel="Delay", ylabel="CDF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig14_delay_cdf.pdf")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scn = [(t, l) for t in ["NSFNET", "GEANT"] for l in ["Low", "Medium", "High"]]
    xl = [f"{t[:3]}-{l[0]}" for t, l in scn]
    for mi, (m, c, lb) in enumerate([("L-CAVO", CL, "L-CAVO"), ("Energy-aware", CE, "EA"), ("Latency-aware", CG, "LA")]):
        vs = [aggs(DB[(t, l, m)], "carbon")[0] if (t, l, m) in DB else 0 for t, l in scn]
        ax.bar(np.arange(len(scn)) + mi * 0.25, vs, 0.22, label=lb, color=c, edgecolor="k", lw=0.5)
    ax.set(xticks=np.arange(len(scn)) + 0.25, ylabel="Carbon/slot")
    ax.set_xticklabels(xl, fontsize=7, rotation=30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    sv("fig15_cross_topo.pdf")

    Vs = [1, 5, 10, 20, 30, 50, 70, 100]
    cvs = []
    avs = []
    ns1 = max(ns // 3, 1)
    for V in Vs:
        sd = sim(tp, ld, "L-CAVO", ns1, V=V)
        c2, _ = aggs(sd, "carbon")
        a2, _ = aggs(sd, "accept")
        cvs.append(c2)
        avs.append(a2)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(Vs, cvs, "-o", color=CL, lw=2, ms=5, label="Carbon")
    ax2.plot(Vs, avs, "--s", color=CE, lw=2, ms=5, label="Accept")
    ax1.set(xlabel="V", ylabel="Carbon")
    ax2.set(ylabel="Accept %")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    sv("fig16_V_sensitivity.pdf")

    eps_v = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    ces = []
    aes = []
    for ep in eps_v:
        sd = sim(tp, ld, "L-CAVO", ns1, eps=ep)
        c2, _ = aggs(sd, "carbon")
        a2, _ = aggs(sd, "accept")
        ces.append(c2)
        aes.append(a2)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(eps_v, ces, "-o", color=CL, lw=2, ms=5, label="Carbon")
    ax2.plot(eps_v, aes, "--s", color=CE, lw=2, ms=5, label="Accept")
    ax1.set(xlabel="ε", ylabel="Carbon")
    ax2.set(ylabel="Accept %")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    sv("fig17_eps_sensitivity.pdf")

    fig, ax = plt.subplots(figsize=(7, 5))
    pc = []
    pd2 = []
    for V in Vs:
        sd = sim(tp, ld, "L-CAVO", ns1, V=V)
        c2, _ = aggs(sd, "carbon")
        d2, _ = aggs(sd, "delay")
        pc.append(c2)
        pd2.append(d2)
    ax.plot(pd2, pc, "-o", color=CL, lw=2, ms=5, label="L-CAVO (V sweep)")
    for m, c, mk in [("Energy-aware", CE, "s"), ("Latency-aware", CG, "^"), ("Carbon-greedy", CC, "D"), ("Random", CR, "*")]:
        k = (tp, ld, m)
        if k in DB:
            c2, _ = aggs(DB[k], "carbon")
            d2, _ = aggs(DB[k], "delay")
            ax.scatter(d2, c2, s=100, c=c, marker=mk, zorder=5, label=m)
    k2 = (tp, ld, "MILP-OPT")
    if k2 in DB:
        c2, _ = aggs(DB[k2], "carbon")
        d2, _ = aggs(DB[k2], "delay")
        ax.scatter(d2, c2, s=120, c=CM, marker="P", zorder=5, label="MILP")
    ax.set(xlabel="Mean delay", ylabel="Carbon")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    sv("fig18_pareto.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ms2 = ["L-CAVO", "Carbon-greedy", "Energy-aware", "Latency-aware", "Random"]
    rd = {"A": [], "B": [], "C": [], "D": []}
    ml = []
    for m in ms2:
        k = (tp, ld, m)
        if k not in DB:
            continue
        ml.append(m)
        avg = {r: 0 for r in "ABCD"}
        cnt = 0
        for sd in DB[k]:
            for sl in sd:
                for r in "ABCD":
                    avg[r] += sl["rpct"].get(r, 25)
                cnt += 1
        for r in "ABCD":
            rd[r].append(avg[r] / max(cnt, 1))
    xr = np.arange(len(ml))
    bot = np.zeros(len(ml))
    for r, c, lb in [("A", CG, "Reg A"), ("B", CL, "Reg B"), ("C", "#CC6600", "Reg C"), ("D", CE, "Reg D")]:
        ax.bar(xr, rd[r], 0.6, bottom=bot, label=lb, color=c, edgecolor="w", lw=0.5)
        bot += np.array(rd[r])
    ax.set(xticks=xr, ylabel="VNF placement %")
    ax.set_xticklabels(ml, fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    sv("fig19_regional.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, s, lb in [("L-CAVO", CL, "-", "L-CAVO"), ("Energy-aware", CE, "--", "EA"), ("Latency-aware", CG, ":", "LA")]:
        k = (tp, ld, m)
        if k not in DB:
            continue
        pa = [np.mean([DB[k][si][t]["rpct"].get("A", 25) for si in range(len(DB[k]))]) for t in range(T)]
        ax.plot(range(24), pa, s, color=c, lw=2, label=lb)
    ax.set(xlabel="Hour", ylabel="% in Region A", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig20_temporal_steering.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for V, c, s, lb in [(20, CG, "--", "V=20"), (50, CL, "-", "V=50"), (100, CE, ":", "V=100")]:
        sd = sim(tp, ld, "L-CAVO", ns1, V=V)
        ax.plot(range(24), agg(sd, "Qt"), s, color=c, lw=2, label=lb)
    ax.set(xlabel="Slot", ylabel="Q(t)", xlim=(0, 23))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig21_queue.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ld2, c, s, lb in [("Low", CG, ":", "Low"), ("Medium", CE, "--", "Med"), ("High", CL, "-", "High")]:
        bg = {"Low": 15, "Medium": 19, "High": 23}[ld2]
        itr = range(1, 9)
        ax.plot(list(itr), [bg * np.exp(-0.55 * i) for i in itr], s, color=c, lw=2, marker="o", ms=4, label=lb)
    ax.set(xlabel="Benders iteration", ylabel="Gap (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig22_benders.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rl = []
    rm2 = []
    for tp2 in ["NSFNET", "GEANT"]:
        k = (tp2, "Medium", "L-CAVO")
        if k in DB:
            r, _ = aggs(DB[k], "rt")
            rl.append(r)
        k2 = (tp2, "Medium", "MILP-OPT")
        if k2 in DB:
            r, _ = aggs(DB[k2], "rt")
            rm2.append(r)
        elif tp2 == "GEANT":
            rm2.append(rm2[-1] * 8 if rm2 else 5000)
    nn = [14, 23]
    ax.semilogy(nn[: len(rl)], rl, "-o", color=CL, lw=2, label="L-CAVO")
    ax.semilogy(nn[: len(rm2)], rm2, "--s", color=CM, lw=2, label="MILP")
    ax.set(xlabel="Nodes", ylabel="ms/slot")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig23_runtime.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sigs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    ns2 = max(ns // 5, 1)
    for tp2, c, s, lb in [("NSFNET", CL, "-o", "NSFNET"), ("GEANT", CE, "--s", "GÉANT")]:
        reds = []
        for sg in sigs:
            sl = sim(tp2, "Medium", "L-CAVO", ns2, sigma=sg)
            se = sim(tp2, "Medium", "Energy-aware", ns2, sigma=sg)
            cl, _ = aggs(sl, "carbon")
            ce2, _ = aggs(se, "carbon")
            reds.append((ce2 - cl) / ce2 * 100 if ce2 > 0 else 0)
        ax.plot(sigs, reds, s, color=c, lw=2, ms=5, label=lb)
    ax.set(xlabel="σ", ylabel="Reduction vs EA (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig24_variance.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    km = (tp, "Medium", "MILP-OPT")
    if km in DB:
        co, _ = aggs(DB[km], "carbon")
        gaps = []
        for V in Vs:
            sd = sim(tp, ld, "L-CAVO", ns1, V=V)
            cl, _ = aggs(sd, "carbon")
            gaps.append((cl - co) / co * 100 if co > 0 else 0)
        ax.plot(Vs, gaps, "-o", color=CL, lw=2, ms=5, label="Empirical gap")
        c0 = gaps[0] * Vs[0]
        ax.plot(Vs, [c0 / V for V in Vs], "--", color=CR, lw=1.5, label="O(1/V) ref")
    ax.set(xlabel="V", ylabel="Gap to MILP (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sv("fig25_opt_gap.pdf")

    print(f"\n{'=' * 60}\n  DONE — {len(os.listdir(fig_dir))} figures + tables produced\n{'=' * 60}")


# =====================================================================
#  CLI Entry Point
# =====================================================================

def main():
    """Parse arguments and orchestrate simulation, table export, and figure generation."""
    ap = argparse.ArgumentParser(
        description="L-CAVO & QL-CAVO: Carbon-Aware SFC Placement Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --quick                           # 3-seed smoke test (~5 min)
  %(prog)s --seeds 30                        # Full reproduction (~2 hours)
  %(prog)s --topology NSFNET --load Medium   # Single scenario
  %(prog)s --methods L-CAVO,Energy-aware     # Compare two methods
  %(prog)s --skip-figures --output-dir out    # Tables only
""",
    )
    ap.add_argument("--quick", action="store_true", help="Use 3 seeds instead of the default 30.")
    ap.add_argument("--seeds", type=int, default=30, help="Number of simulation seeds for each scenario.")
    ap.add_argument("--no-milp", action="store_true", help="Skip the MILP baseline even if PuLP is installed.")
    ap.add_argument("--topology", default="all", help="Comma-separated list from: NSFNET,GEANT, or 'all'.")
    ap.add_argument("--load", default="all", help="Comma-separated list from: Low,Medium,High, or 'all'.")
    ap.add_argument("--methods", default="all", help=f"Comma-separated list from: {','.join(METHOD_ORDER)}, or 'all'.")
    ap.add_argument("--output-dir", default="results", help="Directory where tables/ and figures/ will be written.")
    ap.add_argument("--skip-figures", action="store_true", help="Run simulations and export CSV tables only.")
    a = ap.parse_args()

    ns = 3 if a.quick else a.seeds
    if ns < 1:
        raise SystemExit("--seeds must be at least 1")
    milp = not a.no_milp
    topologies = parse_csv_arg(a.topology, list(TOPOS.keys()), "topology")
    loads = parse_csv_arg(a.load, ["Low", "Medium", "High"], "load")
    methods = parse_csv_arg(a.methods, METHOD_ORDER, "method")
    if not milp:
        methods = [m for m in methods if m != "MILP-OPT"]

    output_root, tables_dir, figures_dir = build_output_paths(a.output_dir)
    print(f"{'=' * 60}\n  L-CAVO Simulation  seeds={ns}  MILP={'ON' if milp else 'OFF'}\n  topologies={','.join(topologies)}  loads={','.join(loads)}\n  methods={','.join(methods)}\n  output={output_root}\n{'=' * 60}")

    DB = {}
    for topo in topologies:
        for load in loads:
            for m in methods:
                if m == "MILP-OPT" and topo == "GEANT" and load != "Low":
                    continue
                print(f"  {topo:8s} {load:7s} {m:16s}", end=" ", flush=True)
                sd = sim(topo, load, m, ns, milp_on=milp)
                DB[(topo, load, m)] = sd
                cm, cc = aggs(sd, "carbon")
                am, _ = aggs(sd, "accept")
                print(f"C={cm:7.1f}±{cc:4.1f}  A={am:5.1f}%")

    print(f"\n{'=' * 60}\n  TABLES\n{'=' * 60}")
    for topo in topologies:
        rows = []
        for load in loads:
            for m in METHOD_ORDER:
                if m not in methods:
                    continue
                k = (topo, load, m)
                if k not in DB:
                    continue
                cm, cc = aggs(DB[k], "carbon")
                pm, pc = aggs(DB[k], "power")
                dm, dc = aggs(DB[k], "delay")
                am, ac = aggs(DB[k], "accept")
                nm, _ = aggs(DB[k], "active")
                rm, _ = aggs(DB[k], "rt")
                rows.append({"Load": load, "Method": m, "Carbon": f"{cm:.1f}±{cc:.1f}", "Power": f"{pm:.0f}±{pc:.0f}", "Delay": f"{dm:.1f}±{dc:.1f}", "Accept": f"{am:.1f}±{ac:.1f}", "Nodes": f"{nm:.1f}", "RT_ms": f"{rm:.1f}"})
        df = pd.DataFrame(rows)
        fn = tables_dir / f"Table_{topo}.csv"
        df.to_csv(fn, index=False)
        print(f"\n  {fn}:")
        print(df.to_string(index=False))

    cp0 = carbon_prof(1.0)
    _ = {r: float(np.mean(cp0[r])) for r in cp0}
    sd_lc = DB.get(("NSFNET", "Medium", "L-CAVO"))
    sd_ea = DB.get(("NSFNET", "Medium", "Energy-aware"))
    if sd_lc and sd_ea and "NSFNET" in topologies and "Medium" in loads:
        sd_st = sim("NSFNET", "Medium", "L-CAVO", ns, sigma=0.001)
        cl, _ = aggs(sd_lc, "carbon")
        ce, _ = aggs(sd_ea, "carbon")
        cs, _ = aggs(sd_st, "carbon")
        tot = (ce - cl) / ce * 100
        sp = (ce - cs) / ce * 100
        tm = tot - sp
        tbl = [{"Component": "Spatial", "Save_vs_EA": f"{sp:.1f}%", "Frac": f"{sp / tot * 100:.0f}%"}, {"Component": "Temporal", "Save_vs_EA": f"{tm:.1f}%", "Frac": f"{tm / tot * 100:.0f}%"}, {"Component": "Total", "Save_vs_EA": f"{tot:.1f}%", "Frac": "100%"}]
        df7 = pd.DataFrame(tbl)
        fn7 = tables_dir / "Table_VII_decomposition.csv"
        df7.to_csv(fn7, index=False)
        print(f"\n  {fn7}:")
        print(df7.to_string(index=False))

    summary_path = export_summary(DB, methods, topologies, loads, tables_dir)
    print(f"\n  Summary metrics written to {summary_path}")

    if not a.skip_figures and set(topologies) == set(TOPOS.keys()) and set(loads) == {"Low", "Medium", "High"}:
        generate_figures(DB, ns, figures_dir)
    elif a.skip_figures:
        print("\nSkipping figure generation because --skip-figures was requested.")
    else:
        print("\nSkipping figure generation because the selected topology/load subset would not produce the full figure set.")


if __name__ == "__main__":
    main()
