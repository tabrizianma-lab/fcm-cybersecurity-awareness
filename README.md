# Replication Package: Modeling the Socio-Technical Dynamics of Cybersecurity Awareness Capability Using Fuzzy Cognitive Maps

This repository contains the complete dataset, simulation code, and analytical outputs required to independently reproduce all empirical findings, reliability statistics, and scenario analyses presented in the manuscript:

> **"Modeling the Socio-Technical Dynamics of Cybersecurity Awareness Capability Using Fuzzy Cognitive Maps"**  
> *Submitted to Scientific Reports*

---

## 🔬 Overview & Purpose

The objective of this repository is to ensure full transparency and computational reproducibility. It provides:
1. **Raw elicitation data** collected from the panel of 9 cybersecurity and human-factor experts.
2. **Reliability routines** demonstrating inter-rater consensus (ICC(2,k)).
3. **FCM state-propagation models** simulating socio-technical dynamics across 7 operational scenarios.
4. **Sensitivity analysis workflows** assessing network stability under parametric perturbations ($\pm 20\%$).

---

## 📁 Repository Contents

All files are provided in the root directory for direct execution:

### 1. Data Files
- `Experts.xlsx`: Anonymized expert panel demographics, credentials, and organizational roles ($N = 9$).
- `weights_by_experts.xlsx`: Raw expert assessment matrices scoring the 34 causal relationships ($R_1$–$R_{34}$) between the 14 socio-technical concepts.
- `icc_results.xlsx`: Statistical outputs of the Intraclass Correlation Coefficient (ICC(2,k)), variance components, and 95% confidence intervals.

### 2. Analytical & Simulation Scripts
- `icc.py`: Computes two-way random-effects intraclass correlation metrics and evaluates inter-rater reliability.
- `fcm_modeling.py`: Executes the core Fuzzy Cognitive Map inference process over 10 discrete propagation steps across 7 defined intervention scenarios; generates steady-state convergence reports and topological graphs.
- `Sensitivity_analysis.py`: Runs local edge sensitivity perturbations ($\pm 20\%$) to identify critical causal pathways and evaluate model robustness.

### 3. Generated Figures & Output Reports
- `fcm_awareness_graph.png` / `fcm_awareness_graph.svg`: High-resolution network diagrams depicting the 14 concepts and 34 weighted causal arcs.
- `fcm_sensitivity_analysis.png`: Two-panel sensitivity curves and edge criticality rankings.
- `fcm_awareness_results.txt`: Numerical activation levels and baseline-comparison deltas for all concepts across all 7 scenarios.
- `fcm_feedback_loop_report.txt`: Analytical breakdown of systemic feedback loops and regulatory cycles identified in the cognitive map.
- `fcm_sensitivity_report.txt`: Tabular summary of sensitivity gradients and threshold responsiveness.

---

## ⚙️ Environment Setup & Dependencies

The simulation framework runs on standard **Python 3.9+**. Install the required dependencies:
```bash
pip install numpy scipy pandas openpyxl matplotlib networkx
