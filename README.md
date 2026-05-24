<div align="center">

<a id="top"></a>

# 🌱 Carbon-Aware Service Function Chain Placement

### ⚡ L-CAVO & QL-CAVO Simulation Framework

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGCN-00629B?style=for-the-badge&logo=ieee&logoColor=white)](#-citation)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-2EA043?style=for-the-badge)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1+-FF6F00?style=for-the-badge)](https://networkx.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9-11557C?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![Tests](https://img.shields.io/badge/Tests-16%20passing-2EA043?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

---

**Carbon-Aware Service Function Chain Placement Under Time-Varying Grid Intensity**
**A Lyapunov-Optimisation and Q-Learning Framework**

_Yassir Al-Karawi_ · _Department of Communications Engineering · University of Diyala, Iraq_

[🔭 Overview](#-overview) ·
[🧠 Algorithms](#-proposed-algorithms) ·
[🚀 Quick Start](#-quick-start) ·
[📊 Results](#-key-results) ·
[🖼️ Figures](#-generated-figures) ·
[📚 Citation](#-citation)

</div>

---

## 🔭 Overview

Modern NFV-enabled networks can substantially reduce operational **carbon emissions** by steering VNF placement toward data-centre regions powered by cleaner energy sources. This repository provides the **complete simulation framework** for reproducing all results in our IEEE TGCN paper.

We propose **two online algorithms** that jointly optimise carbon emissions, SFC acceptance rate, and end-to-end delay — **without requiring future knowledge** of grid carbon intensity:

<div align="center">

| 🏷️ Algorithm | 🧮 Technique | 🌿 Carbon Reduction (vs EA) | ✅ SFC Acceptance |
|:------------:|:------------:|:---------------------------:|:-----------------:|
| **L-CAVO**   | Lyapunov optimisation + Benders decomposition | **26 – 27 %** | 91 – 98 % |
| **QL-CAVO**  | Tabular Q-learning + greedy placement         | **17 – 18 %** | 94 – 99 % |

</div>

> [!NOTE]
> Both algorithms are benchmarked against **six baselines** — MILP-OPT (offline optimal), Energy-aware, Latency-aware, Carbon-greedy, and Random — on two real-world topologies (NSFNET and GÉANT).

---

## 🏗️ System Architecture

<p align="center">
  <img src="figures/architecture.png" width="850" alt="System architecture diagram"/>
</p>

### High-Level Data Flow

```mermaid
flowchart LR
    A[🌐 SFC Request Stream] --> B{Orchestrator}
    C[🔋 Carbon-Intensity Feed] --> B
    D[📡 Topology &<br/>Resource State] --> B
    B --> E[L-CAVO<br/>Lyapunov + Benders]
    B --> F[QL-CAVO<br/>Tabular Q-Learning]
    E --> G[💚 VNF Placement Decision]
    F --> G
    G --> H[📈 Metrics:<br/>CO₂ · Delay · Acceptance]

    classDef green fill:#10B981,stroke:#065F46,color:#fff,stroke-width:2px;
    classDef blue  fill:#3B82F6,stroke:#1E3A8A,color:#fff,stroke-width:2px;
    classDef amber fill:#F59E0B,stroke:#92400E,color:#fff,stroke-width:2px;
    class A,C,D blue;
    class E,F amber;
    class G,H green;
```

---

## 🧠 Proposed Algorithms

### 🟢 L-CAVO — Lyapunov Carbon-Aware VNF Orchestration

L-CAVO transforms the long-term carbon-minimisation problem into a sequence of **per-slot optimisation sub-problems** using **Lyapunov drift-plus-penalty**. A virtual queue `Q(t)` tracks SFC-rejection debt, and the trade-off parameter `V` balances carbon savings against acceptance guarantees.

**Scoring function:**

```math
\text{score}(n) \;=\; V \cdot \Delta\text{CO}_2(n) \;+\; 0.5 \cdot \text{delay} \;+\; 0.3 \cdot \text{utilisation} \;+\; 0.05 \cdot Q(t)
```

> [!TIP]
> As `V` increases, the algorithm prioritises carbon reduction more aggressively. The theoretical optimality gap decreases as **O(1/V)**.

```mermaid
flowchart TD
    S[New time-slot t] --> Q[Read virtual queue Q t]
    Q --> R{SFC requests}
    R -->|For each request| SC[Compute score per node<br/>V · ΔCO₂ + delay + util + Q]
    SC --> BD[Benders decomposition<br/>master + sub-problem]
    BD --> P[Place VNFs greedily<br/>by lowest score]
    P --> UQ[Update Q t+1<br/>by rejection debt]
    UQ --> M[Record metrics]
    M --> S

    classDef start fill:#10B981,stroke:#065F46,color:#fff;
    classDef proc  fill:#3B82F6,stroke:#1E3A8A,color:#fff;
    classDef dec   fill:#F59E0B,stroke:#92400E,color:#fff;
    class S,M start;
    class Q,SC,BD,P,UQ proc;
    class R dec;
```

### 🔵 QL-CAVO — Q-Learning Carbon-Aware VNF Orchestration

QL-CAVO uses a **tabular Q-learning agent** that learns which regions to favour at each time-of-day. The state space encodes `(hour, mean_carbon_intensity, queue_length)`, and actions correspond to region-preference weights.

> [!IMPORTANT]
> **Training:** 300 pre-training episodes on historical carbon profiles, followed by online updates during simulation.

```mermaid
flowchart LR
    OBS[Observe state<br/>hour · CO₂ · queue] --> SEL[ε-greedy action<br/>region weights]
    SEL --> ACT[Greedy placement<br/>under weights]
    ACT --> REW[Reward<br/>= -CO₂ - λ · rejects]
    REW --> UPD[Q-table update<br/>Q ← Q + α · TD]
    UPD --> OBS

    classDef ql fill:#8B5CF6,stroke:#4C1D95,color:#fff,stroke-width:2px;
    class OBS,SEL,ACT,REW,UPD ql;
```

---

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/YassirALKarawi/carbon-aware-sfc.git
cd carbon-aware-sfc
pip install -r requirements.txt
```

### 2️⃣ Run Simulations

```bash
# Quick smoke test (3 seeds, ~5 min)
python lcavo_sim.py --quick

# Full reproduction (30 seeds, ~2 hours)
python lcavo_sim.py --seeds 30

# Generate to a custom output directory
python lcavo_sim.py --output-dir results/full_run
```

### 3️⃣ Reproduce Specific Paper Results

```bash
# Table III — NSFNET results
python lcavo_sim.py --topology NSFNET --seeds 30

# Table IV — GÉANT results
python lcavo_sim.py --topology GEANT --seeds 30

# Fig. 9 — V-sensitivity analysis
python lcavo_sim.py --topology NSFNET --load Medium --methods L-CAVO

# Fig. 10 — ε-sensitivity analysis
python lcavo_sim.py --topology NSFNET --load Medium --methods L-CAVO --skip-figures
```

> [!WARNING]
> Full 30-seed reproduction is computationally intensive (~2 h on a modern workstation). Use `--quick` for a fast end-to-end smoke test before launching the full sweep.

---

## ⚙️ Configuration

| 🔧 Parameter      | 🎚️ Default | 📝 Description |
|:------------------|:----------:|:---------------|
| `--seeds`         | `30`       | Number of independent simulation runs per scenario |
| `--quick`         | _off_      | Shortcut for a 3-seed smoke run |
| `--topology`      | `all`      | Subset of `NSFNET,GEANT` |
| `--load`          | `all`      | Subset of `Low,Medium,High` |
| `--methods`       | `all`      | Subset of supported algorithms and baselines |
| `--output-dir`    | `results`  | Root directory for `tables/` and `figures/` |
| `--skip-figures`  | _off_      | Export CSV tables only (faster) |
| `--no-milp`       | _off_      | Disable MILP baseline even if PuLP is available |

---

## 📊 Key Results

### 🌍 Carbon Reduction Performance (NSFNET)

<div align="center">

| Algorithm         | 🟢 Low Load | 🟡 Medium Load | 🔴 High Load |
|:------------------|:-----------:|:--------------:|:------------:|
| **L-CAVO** vs EA  |  **~27 %**  |   **~26 %**    |  **~26 %**   |
| **QL-CAVO** vs EA |   ~18 %     |    ~17 %       |   ~17 %      |
| Carbon-greedy     |   ~12 %     |    ~11 %       |   ~10 %      |

</div>

### 🌱 Carbon Reduction vs Energy-Aware Baseline

<p align="center">
  <img src="figures/carbon_reduction.png" width="850" alt="Carbon reduction vs EA baseline"/>
</p>

### ⏰ Regional Carbon Intensity Profiles

<p align="center">
  <img src="figures/carbon_profiles.png" width="850" alt="Regional carbon intensity profiles over 24h"/>
</p>

### ✅ SFC Acceptance Rate

<p align="center">
  <img src="figures/acceptance_rate.png" width="850" alt="SFC acceptance rate across loads"/>
</p>

### 🌐 Cross-Topology Comparison

<p align="center">
  <img src="figures/cross_topology.png" width="850" alt="Cross-topology comparison NSFNET vs GEANT"/>
</p>

### 📈 V-Sensitivity Analysis

<p align="center">
  <img src="figures/v_sensitivity.png" width="850" alt="V parameter sensitivity"/>
</p>

### ⏱️ End-to-End Delay CDF

<p align="center">
  <img src="figures/delay_cdf.png" width="850" alt="End-to-end delay CDF"/>
</p>

### 🧩 Decomposition of Carbon Savings

L-CAVO achieves savings through **two complementary mechanisms**:

- 🗺️ **Spatial steering** — placing VNFs in regions with lower instantaneous carbon intensity
- 🕐 **Temporal steering** — shifting workload toward hours when cleaner energy is available

---

## 🖼️ Generated Figures

The full simulation produces **19 publication-quality PDF figures**:

<div align="center">

| #  | 📄 Figure                       | 📝 Description |
|:--:|:--------------------------------|:---------------|
| 7  | `fig07_carbon_profiles.pdf`     | Regional carbon intensity over 24 hours |
| 8  | `fig08_reduction_vs_EA.pdf`     | Carbon reduction vs Energy-aware baseline |
| 9  | `fig09_reduction_vs_LA.pdf`     | Carbon reduction vs Latency-aware baseline |
| 10 | `fig10_hourly_carbon.pdf`       | Hourly carbon emissions by algorithm |
| 11 | `fig11_cumulative.pdf`          | Cumulative carbon over the day |
| 12 | `fig12_active_nodes.pdf`        | Active server count per slot |
| 13 | `fig13_power.pdf`               | Power consumption over time |
| 14 | `fig14_delay_cdf.pdf`           | End-to-end delay CDF |
| 15 | `fig15_cross_topo.pdf`          | Cross-topology carbon comparison |
| 16 | `fig16_V_sensitivity.pdf`       | `V` parameter sensitivity (carbon vs acceptance) |
| 17 | `fig17_eps_sensitivity.pdf`     | `ε` parameter sensitivity |
| 18 | `fig18_pareto.pdf`              | Carbon–delay Pareto frontier |
| 19 | `fig19_regional.pdf`            | Regional VNF placement distribution |
| 20 | `fig20_temporal_steering.pdf`   | Temporal steering toward clean regions |
| 21 | `fig21_queue.pdf`               | Virtual queue `Q(t)` evolution |
| 22 | `fig22_benders.pdf`             | Benders decomposition convergence |
| 23 | `fig23_runtime.pdf`             | Runtime scaling (L-CAVO vs MILP) |
| 24 | `fig24_variance.pdf`            | Sensitivity to carbon-intensity variance σ |
| 25 | `fig25_opt_gap.pdf`             | Optimality gap vs `V` (O(1/V) bound) |

</div>

---

## 📁 Repository Structure

```text
carbon-aware-sfc/
├── 📜 lcavo_sim.py          # Main simulation engine (all algorithms & figure generation)
├── 📦 requirements.txt      # Python dependencies
├── 📄 LICENSE               # MIT License
├── 📘 README.md             # This file
├── 🖼️  figures/              # Pre-rendered figures used in the README
├── 🧪 tests/                # Unit tests (16 tests, run via pytest)
└── 📊 results/              # Generated outputs (created on first run)
    ├── tables/              #   CSV summary tables
    └── figures/             #   Publication-quality PDF figures
```

---

## 📦 Requirements

<div align="center">

| Package        | Version    | Purpose |
|:---------------|:----------:|:--------|
| **Python**     | ≥ 3.9      | Runtime |
| **NumPy**      | 1.26       | Numerics |
| **SciPy**      | 1.13       | Statistics |
| **Pandas**     | 2.2        | Tables & CSV export |
| **NetworkX**   | ≥ 3.1      | Topology graphs |
| **Matplotlib** | 3.9        | Publication figures |
| **PuLP**       | 2.7        | MILP baseline |

</div>

> [!TIP]
> **Optional:** [Gurobi 11.0](https://www.gurobi.com/academia/academic-program-and-licenses/) (free academic license) can replace CBC for substantially faster MILP benchmarks.

---

## 🧪 Testing

```bash
# Run the full test suite (16 tests)
pytest -q

# Run a specific suite
pytest tests/test_algorithms.py -v
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{alqaisy2026carbon,
  author  = {Al-Karawi, Yassir},
  title   = {Carbon-Aware Service Function Chain Placement Under
             Time-Varying Grid Intensity: A {Lyapunov}-Optimisation
             and {Q}-Learning Framework},
  journal = {IEEE Trans. Green Commun. Netw.},
  year    = {2026},
  note    = {Submitted}
}
```

---

<div align="center">

### 📜 License & Attribution

**MIT License** _(simulation code)_ — see [LICENSE](LICENSE) for details.<br/>
The manuscript is © 2026 the author and may not be redistributed without permission.

---

**Made with 💚 at the University of Diyala**

[⬆ Back to top](#top)

</div>
