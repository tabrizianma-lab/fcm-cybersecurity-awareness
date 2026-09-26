import numpy as np

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

idx = {name: i for i, name in enumerate(concepts)}
n = len(concepts)

edges = {
    "R1": ("Attack Surface Complexity", "Threat Exposure", 0.5),
    "R2": ("Threat Exposure", "Organizational Cyber Risk", 0.805),
    "R3": ("Attack Surface Complexity", "Organizational Cyber Risk", 0.722),
    "R4": ("Governance Maturity", "Cybersecurity Investment and Resource Allocation", 0.611),
    "R5": ("Governance Maturity", "Preventive Control Strength", 0.527),
    "R6": ("Governance Maturity", "Monitoring and Detection Capability", 0.444),
    "R7": ("Governance Maturity", "Incident Response Readiness", 0.5),
    "R8": ("Governance Maturity", "Cybersecurity Awareness Capability", 0.805),
    "R9": ("Cybersecurity Investment and Resource Allocation", "Preventive Control Strength", 0.583),
    "R10": ("Cybersecurity Investment and Resource Allocation", "Monitoring and Detection Capability", 0.472),
    "R11": ("Cybersecurity Investment and Resource Allocation", "Incident Response Readiness", 0.555),
    "R12": ("Cybersecurity Investment and Resource Allocation", "Cybersecurity Awareness Capability", 0.611),
    "R13": ("Threat Exposure", "Monitoring and Detection Capability", 0.305),
    "R14": ("Threat Exposure", "Incident Response Readiness", 0.25),
    "R15": ("Threat Exposure", "Cybersecurity Awareness Capability", 0.222),
    "R16": ("Cybersecurity Investment and Resource Allocation", "Enacted Control Effectiveness", 0.472),
    "R17": ("Preventive Control Strength", "Monitoring and Detection Capability", 0.416),
    "R18": ("Preventive Control Strength", "Incident Response Readiness", 0.416),
    "R19": ("Preventive Control Strength", "Enacted Control Effectiveness", 0.555),
    "R20": ("Monitoring and Detection Capability", "Incident Response Readiness", 0.667),
    "R21": ("Monitoring and Detection Capability", "Enacted Control Effectiveness", 0.472),
    "R22": ("Incident Response Readiness", "Enacted Control Effectiveness", 0.388),
    "R23": ("Cybersecurity Awareness Capability", "Secure Compliance Behavior", 0.694),
    "R24": ("Cybersecurity Awareness Capability", "Enacted Control Effectiveness", 0.527),
    "R25": ("Security Culture", "Secure Compliance Behavior", 0.805),
    "R26": ("Security Culture", "Enacted Control Effectiveness", 0.75),
    "R27": ("Secure Compliance Behavior", "Enacted Control Effectiveness", 0.638),
    "R28": ("Workaround Tendency", "Enacted Control Effectiveness", -0.667),
    "R29": ("Preventive Control Strength", "Organizational Cyber Risk", -0.611),
    "R30": ("Monitoring and Detection Capability", "Organizational Cyber Risk", -0.555),
    "R31": ("Incident Response Readiness", "Organizational Cyber Risk", -0.416),
    "R32": ("Enacted Control Effectiveness", "Organizational Cyber Risk", -0.555),
    "R33": ("Automated Risk Modeling and Analytical Support", "Monitoring and Detection Capability", 0.472),
    "R34": ("Automated Risk Modeling and Analytical Support", "Incident Response Readiness", 0.555)
}

W = np.zeros((n, n))

for source, target, weight in edges.values():
    W[idx[source], idx[target]] = weight


scenarios = {
    "Baseline": {
        "description": "No awareness-related intervention; reference condition.",
        "shocks": {}
    },
    "Awareness Campaign": {
        "description": "Direct improvement in cybersecurity awareness capability.",
        "shocks": {
            "Cybersecurity Awareness Capability": 0.1
        }
    },
    "Governance-Driven Awareness": {
        "description": "Improvement in governance maturity indirectly strengthens awareness and security outcomes.",
        "shocks": {
            "Governance Maturity": 0.1
        }
    },
    "Awareness Through Investment": {
        "description": "Investment in training, awareness programs, and security resources.",
        "shocks": {
            "Cybersecurity Investment and Resource Allocation": 0.1,
            "Cybersecurity Awareness Capability": 0.06
        }
    },
    "Threat-Triggered Awareness": {
        "description": "A threat increase is accompanied by an organizational awareness response.",
        "shocks": {
            "Threat Exposure": 0.08,
            "Cybersecurity Awareness Capability": 0.01
        }
    },
    "Awareness Fatigue": {
        "description": "Decline in awareness capability and increase in employee workaround tendency.",
        "shocks": {
            "Cybersecurity Awareness Capability": -0.1,
            "Workaround Tendency": 0.06
        }
    },
    "Awareness and Security Culture": {
        "description": "Combined improvement in awareness capability and security culture.",
        "shocks": {
            "Cybersecurity Awareness Capability": 0.1,
            "Security Culture": 0.08
        } 
    }
}


def simulate(shocks, steps):
    state = np.full(n, 0.50)
    delta = np.zeros(n)

    for concept, value in shocks.items():
        delta[idx[concept]] = value

    for _ in range(steps):
        new_state = np.clip(state + delta, LOWER, UPPER)
        realized_delta = new_state - state
        delta = W.T @ realized_delta
        state = new_state

    return state


def format_number(value):
    return f"{value:.4f}"


results = {}
for scenario_name, scenario_data in scenarios.items():
    results[scenario_name] = simulate(
        scenario_data["shocks"],
        N_STEPS
    )

baseline = results["Baseline"]

with open("fcm_awareness_results.txt", "w", encoding="utf-8") as file:
    file.write("DELTA-BASED FCM SIMULATION REPORT\n")
    file.write("=" * 80 + "\n")
    file.write(f"Number of propagation cycles: {N_STEPS}\n")
    file.write("Initial value of all concepts: 0.5000\n")
    file.write("Value range: 0.0000 to 1.0000\n")
    file.write("=" * 80 + "\n\n")

    for scenario_name, scenario_data in scenarios.items():
        state = results[scenario_name]
        difference = state - baseline

        file.write(f"SCENARIO: {scenario_name}\n")
        file.write(f"Meaning: {scenario_data['description']}\n")
        file.write("-" * 80 + "\n")
        file.write(
            f"{'Concept':<55}"
            f"{'Final Value':>15}"
            f"{'Change vs Baseline':>22}\n"
        )
        file.write("-" * 80 + "\n")

        for i, concept in enumerate(concepts):
            file.write(
                f"{concept:<55}"
                f"{format_number(state[i]):>15}"
                f"{format_number(difference[i]):>22}\n"
            )

        file.write("\n")
        file.write(
            "Key outcome changes:\n"
            f"Organizational Cyber Risk: "
            f"{format_number(state[idx['Organizational Cyber Risk']])} "
            f"({format_number(difference[idx['Organizational Cyber Risk']])} vs baseline)\n"
            f"Enacted Control Effectiveness: "
            f"{format_number(state[idx['Enacted Control Effectiveness']])} "
            f"({format_number(difference[idx['Enacted Control Effectiveness']])} vs baseline)\n"
            f"Cybersecurity Awareness Capability: "
            f"{format_number(state[idx['Cybersecurity Awareness Capability']])} "
            f"({format_number(difference[idx['Cybersecurity Awareness Capability']])} vs baseline)\n"
        )
        file.write("\n" + "=" * 80 + "\n\n")

risk = "Organizational Cyber Risk"
control = "Enacted Control Effectiveness"
awareness = "Cybersecurity Awareness Capability"

tests = {
    "Awareness Campaign increases awareness":
        results["Awareness Campaign"][idx[awareness]] > baseline[idx[awareness]],

    "Governance-Driven Awareness increases awareness":
        results["Governance-Driven Awareness"][idx[awareness]] > baseline[idx[awareness]],

    "Awareness Through Investment improves control effectiveness":
        results["Awareness Through Investment"][idx[control]] > baseline[idx[control]],

    "Threat-Triggered Awareness increases awareness":
        results["Threat-Triggered Awareness"][idx[awareness]] > baseline[idx[awareness]],

    "Awareness Fatigue reduces control effectiveness":
        results["Awareness Fatigue"][idx[control]] < baseline[idx[control]],

    "Awareness and Security Culture improves compliance":
        results["Awareness and Security Culture"][
            idx["Secure Compliance Behavior"]
        ] > baseline[idx["Secure Compliance Behavior"]]
}

with open("fcm_awareness_results.txt", "a", encoding="utf-8") as file:
    file.write("SCENARIO LOGIC TESTS\n")
    file.write("=" * 80 + "\n")

    for test_name, passed in tests.items():
        status = "PASSED" if passed else "FAILED"
        file.write(f"{status} | {test_name}\n")

    file.write("\n")
    file.write("RISK AND CONTROL COMPARISON\n")
    file.write("=" * 80 + "\n")
    file.write(
        f"{'Scenario':<35}"
        f"{'Cyber Risk':>18}"
        f"{'Control Effectiveness':>25}\n"
    )
    file.write("-" * 80 + "\n")

    for scenario_name, state in results.items():
        file.write(
            f"{scenario_name:<35}"
            f"{format_number(state[idx[risk]]):>18}"
            f"{format_number(state[idx[control]]):>25}\n"
        )

print("Simulation completed.")
print(f"Propagation cycles: {N_STEPS}")
print("Output file: fcm_awareness_results.txt")
print("\nScenario tests:")

for test_name, passed in tests.items():
    status = "PASSED" if passed else "FAILED"
    print(f"{status} | {test_name}")

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

short_names = {
    "Attack Surface Complexity": "ASC",
    "Threat Exposure": "TE",
    "Organizational Cyber Risk": "OCR",
    "Governance Maturity": "GM",
    "Cybersecurity Investment and Resource Allocation": "CIR",
    "Preventive Control Strength": "PCS",
    "Monitoring and Detection Capability": "MDC",
    "Incident Response Readiness": "IRR",
    "Cybersecurity Awareness Capability": "CAC",
    "Enacted Control Effectiveness": "ECE",
    "Security Culture": "SC",
    "Secure Compliance Behavior": "SCB",
    "Workaround Tendency": "WT",
    "Automated Risk Modeling and Analytical Support": "ARM"
}

graph = nx.DiGraph()
graph.add_nodes_from(concepts)

for relation_id, (source, target, weight) in edges.items():
    graph.add_edge(
        source,
        target,
        relation=relation_id,
        weight=weight
    )

pos = {
    "Attack Surface Complexity": (3.6, 4.8),
    "Governance Maturity": (-5.0, 0.5),
    "Security Culture": (1, -2),
    "Workaround Tendency": (3.2, 3.0),
    "Automated Risk Modeling and Analytical Support": (-0.5, 1.25),

    "Threat Exposure": (-2.5, 4.0),
    "Cybersecurity Investment and Resource Allocation": (-3.6, 2.3),
    "Cybersecurity Awareness Capability": (-2, -0.5),
    "Preventive Control Strength": (-2.5, -3.2),

    "Monitoring and Detection Capability": (0.3, 3),
    "Incident Response Readiness": (0.3, -0.3),
    "Secure Compliance Behavior": (0.3, -2),

    "Enacted Control Effectiveness": (3.2, 1.0),
    "Organizational Cyber Risk": (3.2, -2.0)
}

node_colors = []

for concept in concepts:
    if concept == "Cybersecurity Awareness Capability":
        node_colors.append("#FFD166")
    elif concept == "Organizational Cyber Risk":
        node_colors.append("#EF476F")
    elif concept in [
        "Enacted Control Effectiveness",
        "Secure Compliance Behavior",
        "Security Culture"
    ]:
        node_colors.append("#06D6A0")
    else:
        node_colors.append("#90CAF9")

plt.figure(figsize=(30, 20))

nx.draw_networkx_nodes(
    graph,
    pos,
    node_color=node_colors,
    node_size=2000,
    edgecolors="#222222",
    linewidths=1.2,
    alpha=0.95
)

nx.draw_networkx_labels(
    graph,
    pos,
    labels=short_names,
    font_size=18,
    font_weight="bold",
    font_color="#111111"
)

for i, (source, target, data) in enumerate(graph.edges(data=True)):
    weight = data["weight"]
    color = "#1976D2" if weight >= 0 else "#D32F2F"
    style = "solid" if weight >= 0 else "solid"
    rad = 0.0 if i % 2 == 0 else -0.0

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(source, target)],
        edge_color=color,
        style=style,
        width=0.8 if weight >= 0 else 0.8,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        node_size=3000,
        connectionstyle=f"arc3,rad={rad}",
        min_source_margin=18,
        min_target_margin=24
    )

edge_labels = {
    (source, target): f"{data['relation']}\n{data['weight']:+.2f}"
    for source, target, data in graph.edges(data=True)
}

nx.draw_networkx_edge_labels(
    graph,
    pos,
    edge_labels=edge_labels,
    font_size=15,
    font_color="#000000",
    rotate=False,
    label_pos=0.2,
    bbox={
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85,
        "pad": 0.25
    }
)

concept_key = "\n".join(
    f"{short_names[concept]} = {concept}"
    for concept in concepts
)

plt.figtext(
    0.02,
    0.015,
    concept_key,
    ha="left",
    va="bottom",
    fontsize=9,
    bbox={
        "facecolor": "white",
        "edgecolor": "#AAAAAA",
        "alpha": 0.95,
        "pad": 0.6
    }
)

plt.title(
    "Cybersecurity Awareness-Centered Fuzzy Cognitive Map",
    fontsize=20,
    fontweight="bold",
    pad=25
)

plt.xlim(-6.2, 4.5)
plt.ylim(-5.2, 5.2)
plt.axis("off")
plt.tight_layout(rect=[0, 0.13, 1, 1])

plt.savefig(
    "fcm_awareness_graph.png",
    dpi=1000,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "fcm_awareness_graph.svg",
    bbox_inches="tight",
    facecolor="white"
)

#plt.show()


import networkx as nx

def canonical_cycle(cycle):
    c = list(cycle)
    r = c[:]
    best = None
    for seq in (c, list(reversed(c))):
        for i in range(len(seq)):
            rot = tuple(seq[i:] + seq[:i])
            if best is None or rot < best:
                best = rot
    return best

def analyze_feedback_loops(edges):
    G = nx.DiGraph()
    for rid, (s, t, w) in edges.items():
        G.add_edge(s, t, relation=rid, weight=w)

    seen = set()
    loops = []

    for cycle in nx.simple_cycles(G):
        key = canonical_cycle(cycle)
        if key in seen:
            continue
        seen.add(key)

        members = list(cycle)
        edge_list = []
        strength = 1.0

        for i in range(len(members)):
            s = members[i]
            t = members[(i + 1) % len(members)]
            data = G[s][t]
            w = float(data["weight"])
            strength *= w
            edge_list.append((data["relation"], s, t, w))

        loop_type = "reinforcing" if strength > 0 else "balancing"
        loops.append({
            "members": members,
            "edges": edge_list,
            "strength": strength,
            "type": loop_type
        })

    loops.sort(key=lambda x: (len(x["members"]), -abs(x["strength"])))
    return loops

loops = analyze_feedback_loops(edges)

lines = []
lines.append("FEEDBACK LOOP REPORT")
lines.append("=" * 80)
lines.append(f"Total loops found: {len(loops)}")
lines.append("")

for i, loop in enumerate(loops, 1):
    lines.append(f"Loop {i}")
    lines.append(f"Type: {loop['type']}")
    lines.append(f"Members: {' -> '.join(loop['members'])} -> {loop['members'][0]}")
    lines.append("Edges:")
    for rid, s, t, w in loop["edges"]:
        lines.append(f"  {rid}: {s} -> {t} ({w:+.2f})")
    lines.append(f"Loop strength: {loop['strength']:+.6f}")
    lines.append("-" * 80)

report = "\n".join(lines)

with open("fcm_feedback_loop_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("\nReport saved to fcm_feedback_loop_report.txt")
