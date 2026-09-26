import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & BASELINE MODEL
# ==========================================
N_STEPS = 10
LOWER = 0.0
UPPER = 1.0

concepts = [
    "Attack Surface Complexity",
    "Threat Exposure",
    "Organizational Cyber Risk",
    "Governance Maturity",
    "Cybersecurity Investment and Resource Allocation",
    "Preventive Control Strength",
    "Monitoring and Detection Capability",
    "Incident Response Readiness",
    "Cybersecurity Awareness Capability",
    "Enacted Control Effectiveness",
    "Security Culture",
    "Secure Compliance Behavior",
    "Workaround Tendency",
    "Automated Risk Modeling and Analytical Support"
]

idx = {c: i for i, c in enumerate(concepts)}

edges = [
    ("Attack Surface Complexity", "Threat Exposure", 0.555),
    ("Threat Exposure", "Organizational Cyber Risk", 0.805),
    ("Governance Maturity", "Cybersecurity Investment and Resource Allocation", 0.777),
    ("Cybersecurity Investment and Resource Allocation", "Preventive Control Strength", 0.750),
    ("Cybersecurity Investment and Resource Allocation", "Monitoring and Detection Capability", 0.777),
    ("Cybersecurity Investment and Resource Allocation", "Incident Response Readiness", 0.777),
    ("Cybersecurity Investment and Resource Allocation", "Cybersecurity Awareness Capability", 0.694),
    ("Governance Maturity", "Cybersecurity Awareness Capability", 0.805),
    ("Governance Maturity", "Security Culture", 0.750),
    ("Governance Maturity", "Automated Risk Modeling and Analytical Support", 0.694),
    ("Automated Risk Modeling and Analytical Support", "Threat Exposure", 0.638),
    ("Automated Risk Modeling and Analytical Support", "Preventive Control Strength", 0.638),
    ("Automated Risk Modeling and Analytical Support", "Monitoring and Detection Capability", 0.694),
    ("Automated Risk Modeling and Analytical Support", "Incident Response Readiness", 0.694),
    ("Automated Risk Modeling and Analytical Support", "Cybersecurity Awareness Capability", 0.611),
    ("Cybersecurity Awareness Capability", "Security Culture", 0.777),
    ("Cybersecurity Awareness Capability", "Secure Compliance Behavior", 0.805),
    ("Cybersecurity Awareness Capability", "Workaround Tendency", -0.722),
    ("Cybersecurity Awareness Capability", "Enacted Control Effectiveness", 0.722),
    ("Preventive Control Strength", "Enacted Control Effectiveness", 0.722),
    ("Monitoring and Detection Capability", "Incident Response Readiness", 0.694),
    ("Incident Response Readiness", "Organizational Cyber Risk", -0.555),
    ("Incident Response Readiness", "Enacted Control Effectiveness", 0.583),
    ("Security Culture", "Cybersecurity Awareness Capability", 0.694),
    ("Security Culture", "Secure Compliance Behavior", 0.805),
    ("Security Culture", "Workaround Tendency", -0.750),
    ("Secure Compliance Behavior", "Enacted Control Effectiveness", 0.777),
    ("Workaround Tendency", "Enacted Control Effectiveness", -0.667),
    ("Preventive Control Strength", "Organizational Cyber Risk", -0.611),
    ("Monitoring and Detection Capability", "Organizational Cyber Risk", -0.527),
    ("Enacted Control Effectiveness", "Attack Surface Complexity", -0.500),
    ("Enacted Control Effectiveness", "Organizational Cyber Risk", -0.555),
    ("Enacted Control Effectiveness", "Threat Exposure", -0.527),
    ("Automated Risk Modeling and Analytical Support", "Organizational Cyber Risk", -0.500),
]

def build_weight_matrix(edge_list):
    W = np.zeros((len(concepts), len(concepts)))
    for s, t, w in edge_list:
        W[idx[s], idx[t]] = w
    return W

W_base = build_weight_matrix(edges)

scenarios = {
    "Baseline": {},
    "Awareness Campaign": {"Cybersecurity Awareness Capability": 0.10},
    "Governance-Driven Awareness": {"Governance Maturity": 0.10},
    "Awareness Through Investment": {"Cybersecurity Investment and Resource Allocation": 0.10, "Cybersecurity Awareness Capability": 0.06},
    "Threat-Triggered Awareness": {"Threat Exposure": 0.08, "Cybersecurity Awareness Capability": 0.01},
    "Awareness Fatigue": {"Cybersecurity Awareness Capability": -0.10, "Workaround Tendency": 0.06},
    "Awareness & Culture Synergy": {"Cybersecurity Awareness Capability": 0.10, "Security Culture": 0.08},
}

def simulate(W_mat, scenario_shocks, steps=N_STEPS, init_val=0.50):
    state = np.full(len(concepts), init_val)
    delta = np.zeros(len(concepts))
    for c_name, val in scenario_shocks.items():
        delta[idx[c_name]] = val

    for _ in range(steps):
        new_state = np.clip(state + delta, LOWER, UPPER)
        realized_delta = new_state - state
        delta = W_mat.T @ realized_delta
        state = new_state
    return state

# ==========================================
# 1. BASELINE EXECUTION & SCENARIO RESULTS
# ==========================================
baseline_final = simulate(W_base, scenarios["Baseline"])
results = {}
for name, shocks in scenarios.items():
    results[name] = simulate(W_base, shocks)

with open("fcm_awareness_results.txt", "w", encoding="utf-8") as f:
    f.write("="*75 + "\n")
    f.write("FCM SIMULATION REPORT (CAC FRAMEWORK)\n")
    f.write("="*75 + "\n\n")
    for s_name, final_s in results.items():
        diff = final_s - baseline_final
        f.write(f"--- Scenario: {s_name} ---\n")
        for i, c in enumerate(concepts):
            f.write(f"{c:<50} | Final: {final_s[i]:.4f} | Diff: {diff[i]:+.4f}\n")
        f.write("\n")

# ==========================================
# 2. SENSITIVITY ANALYSIS MODULE
# ==========================================
def run_sensitivity_analysis():
    print("Running Sensitivity Analysis...")
    report_lines = []
    report_lines.append("="*75)
    report_lines.append("FCM SENSITIVITY ANALYSIS & ROBUSTNESS AUDIT")
    report_lines.append("="*75 + "\n")

    # --- 2.1 One-At-A-Time (OAT) Edge Sensitivity (±20%) ---
    report_lines.append("[1] ONE-AT-A-TIME (OAT) EDGE SENSITIVITY (±20% Weight Perturbation)")
    report_lines.append("-" * 75)
    
    ocr_idx = idx["Organizational Cyber Risk"]
    ece_idx = idx["Enacted Control Effectiveness"]
    cac_idx = idx["Cybersecurity Awareness Capability"]

    edge_sensitivities = []
    
    for i, (u, v, w) in enumerate(edges):
        # +20%
        edges_plus = list(edges)
        edges_plus[i] = (u, v, np.clip(w * 1.20, -1.0, 1.0))
        W_plus = build_weight_matrix(edges_plus)
        res_plus = simulate(W_plus, scenarios["Awareness & Culture Synergy"])
        
        # -20%
        edges_minus = list(edges)
        edges_minus[i] = (u, v, np.clip(w * 0.80, -1.0, 1.0))
        W_minus = build_weight_matrix(edges_minus)
        res_minus = simulate(W_minus, scenarios["Awareness & Culture Synergy"])
        
        # Base S6 result
        base_res = results["Awareness & Culture Synergy"]
        
        max_dev = max(np.max(np.abs(res_plus - base_res)), np.max(np.abs(res_minus - base_res)))
        ocr_dev = max(abs(res_plus[ocr_idx] - base_res[ocr_idx]), abs(res_minus[ocr_idx] - base_res[ocr_idx]))
        
        edge_sensitivities.append({
            "edge": f"R{i+1}: {u} -> {v}",
            "base_w": w,
            "max_system_dev": max_dev,
            "ocr_dev": ocr_dev
        })

    # Sort critical edges
    edge_sensitivities.sort(key=lambda x: x["max_system_dev"], reverse=True)
    
    report_lines.append(f"{'Edge Identifier / Relationship':<55} | {'Base W':<7} | {'Max Dev':<8} | {'OCR Dev':<8}")
    report_lines.append("-" * 85)
    for row in edge_sensitivities[:10]:
        report_lines.append(f"{row['edge']:<55} | {row['base_w']:<+7.3f} | {row['max_system_dev']:<8.4f} | {row['ocr_dev']:<8.4f}")
    report_lines.append("\nTop 5 critical causal paths identify the primary drivers of model output variance.\n")

    # --- 2.2 Global Monte Carlo Weight Perturbation (±25% Uniform Noise) ---
    report_lines.append("[2] GLOBAL STRUCTURAL PERTURBATION (Monte Carlo, N=200, Noise=±25%)")
    report_lines.append("-" * 75)
    
    mc_runs = 200
    noise_level = 0.25
    mc_results = []
    
    for _ in range(mc_runs):
        perturbed_edges = []
        for u, v, w in edges:
            noise = np.random.uniform(-noise_level, noise_level)
            new_w = np.clip(w * (1 + noise), -1.0, 1.0)
            perturbed_edges.append((u, v, new_w))
        
        W_pert = build_weight_matrix(perturbed_edges)
        res_pert = simulate(W_pert, scenarios["Awareness & Culture Synergy"])
        mc_results.append(res_pert)
        
    mc_results = np.array(mc_results)
    mean_state = np.mean(mc_results, axis=0)
    std_state = np.std(mc_results, axis=0)
    cv_state = (std_state / mean_state) * 100 # Coefficient of Variation (%)
    
    report_lines.append(f"{'Concept Name':<45} | {'Mean State':<10} | {'Std Dev':<10} | {'CV (%)':<8}")
    report_lines.append("-" * 80)
    for i, c in enumerate(concepts):
        report_lines.append(f"{c:<45} | {mean_state[i]:<10.4f} | {std_state[i]:<10.4f} | {cv_state[i]:<8.2f}%")
        
    avg_cv = np.mean(cv_state)
    report_lines.append(f"\nAverage Structural Coefficient of Variation across all concepts: {avg_cv:.2f}% (Threshold < 5% confirms high robustness).")

    # --- 2.3 Scenario Intensity Sweep (S6: Synergistic Sensitivity) ---
    report_lines.append("\n[3] SCENARIO INPUT INTENSITY SWEEP (S6 Multiplier from 0.2x to 2.5x)")
    report_lines.append("-" * 75)
    
    multipliers = np.linspace(0.2, 2.5, 15)
    sweep_cac_inputs = []
    sweep_ocr = []
    sweep_ece = []
    sweep_cac_out = []
    
    for m in multipliers:
        shocks = {
            "Cybersecurity Awareness Capability": 0.10 * m,
            "Security Culture": 0.08 * m
        }
        res = simulate(W_base, shocks)
        sweep_cac_inputs.append(0.10 * m)
        sweep_ocr.append(res[ocr_idx])
        sweep_ece.append(res[ece_idx])
        sweep_cac_out.append(res[cac_idx])
        
    report_lines.append(f"{'Stimulus (CAC shock)':<22} | {'CAC Output':<12} | {'ECE Output':<12} | {'OCR (Risk)':<12}")
    report_lines.append("-" * 65)
    for k in range(len(multipliers)):
        report_lines.append(f"{sweep_cac_inputs[k]:<22.3f} | {sweep_cac_out[k]:<12.4f} | {sweep_ece[k]:<12.4f} | {sweep_ocr[k]:<12.4f}")

    # Write report
    with open("fcm_sensitivity_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # --- Plot Sensitivity Charts ---
    plt.figure(figsize=(12, 5), dpi=300)
    
    # Subplot 1: Dynamic Response Sweep
    plt.subplot(1, 2, 1)
    plt.plot(sweep_cac_inputs, sweep_ocr, 'r-o', label='Cyber Risk (OCR)', linewidth=2)
    plt.plot(sweep_cac_inputs, sweep_ece, 'g-s', label='Control Effectiveness (ECE)', linewidth=2)
    plt.plot(sweep_cac_inputs, sweep_cac_out, 'b-^', label='Awareness (CAC)', linewidth=2)
    plt.title('Non-Linear Response to S6 Intervention Intensity', fontsize=11, fontweight='bold')
    plt.xlabel('Initial CAC Shock Magnitude (Δ)', fontsize=10)
    plt.ylabel('Final Concept Activation', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)

    # Subplot 2: Edge Criticality Ranking
    plt.subplot(1, 2, 2)
    top_edges = edge_sensitivities[:8]
    edge_names = [e["edge"].split(":")[0] for e in top_edges]
    edge_impacts = [e["max_system_dev"] for e in top_edges]
    y_pos = np.arange(len(edge_names))
    
    plt.barh(y_pos, edge_impacts, color='#1976D2', align='center', alpha=0.85)
    plt.yticks(y_pos, edge_names)
    plt.gca().invert_yaxis()
    plt.title('Top 8 Critical Causal Edges (Max Shift on ±20%)', fontsize=11, fontweight='bold')
    plt.xlabel('Maximum Concept Deviation', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("fcm_sensitivity_analysis.png")
    plt.close()
    print("Sensitivity Analysis complete. Outputs saved: 'fcm_sensitivity_report.txt' and 'fcm_sensitivity_analysis.png'.")

# Run analysis
run_sensitivity_analysis()
