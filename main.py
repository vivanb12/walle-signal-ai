import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from environment import GridWorldEnv
from agent import WallEAgent

def run_mission(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    env = GridWorldEnv(config)
    agent = WallEAgent(config)
    
    path = [env.agent_pos.copy()]
    entropy_history = []
    trace = []

    print("\n--- WALL-E MISSION SIMULATION ENGAGED ---")
    for step in range(config["max_steps"]):
        action, target, rationale = agent.plan_next_action(env.agent_pos, env.battery, step)
        success, obs = env.step(action, target)

        if not success:
            print(f"[MISSION ABORTED] Step {step:02d}: {obs['error']}")
            break

        if "scan" in obs:
            agent.belief.update_scan(obs["position"], obs["scan"])
        if "clue" in obs:
            agent.belief.update_clue(obs["clue"])

        entropy = agent.belief.get_entropy()
        entropy_history.append(entropy)
        path.append(env.agent_pos.copy())

        step_record = {
            "step": int(step),
            "action": action,
            "target": [int(x) for x in target] if isinstance(target, (np.ndarray, list)) else target,
            "pos": [int(x) for x in obs["position"]],
            "battery": float(round(obs["battery"], 2)),
            "entropy": float(round(entropy, 3)),
            "rationale": rationale
        }
        trace.append(step_record)

        print(f"Step {step:02d} | {action:<8} | Pos: {[int(x) for x in obs['position']]} | Batt: {obs['battery']:5.1f}% | Entropy: {entropy:5.2f} | {rationale}")

        if np.array_equal(env.agent_pos, env.eve_pos):
            print(f"\n[TARGET LOCATED] WALL-E successfully reached EVE at {env.eve_pos.tolist()} in {step} steps.")
            break

    # Build report with native Python types for JSON compatibility
    best_loc, confidence = agent.belief.get_mode()
    report = {
        "final_inferred_location": [int(x) for x in best_loc],
        "true_eve_location": [int(x) for x in config["eve_true_location"]],
        "confidence_score": float(round(confidence, 4)),
        "total_steps": int(len(trace)),
        "clues_discovered": int(len(env.discovered_clues)),
        "final_battery": float(round(env.battery, 2)),
        "target_reached": bool(np.array_equal(env.agent_pos, env.eve_pos))
    }

    with open("mission_trace.json", "w") as f:
        json.dump(trace, f, indent=2)

    with open("mission_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- FINAL MISSION REPORT ---")
    print(json.dumps(report, indent=2))

    render_visualizations(path, agent.belief.grid, entropy_history, config)

def render_visualizations(path, belief_grid, entropy_hist, config):
    path = np.array(path)
    fig = plt.figure(figsize=(18, 5))

    # Subplot 1: 3D Surface & Path
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    X, Y = np.meshgrid(np.arange(config["grid_size"]), np.arange(config["grid_size"]))
    Z = belief_grid.T
    ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.45, edgecolor='none')

    z_path = np.linspace(0.005, np.max(Z) * 1.05, len(path))
    ax1.plot(path[:, 0], path[:, 1], z_path, label="WALL-E Path", color='cyan', linewidth=2.5, marker='o', markersize=3)
    ax1.scatter(config["eve_true_location"][0], config["eve_true_location"][1], np.max(Z), color='red', s=90, label="EVE (Target)")
    ax1.set_title("3D Belief Distribution & Path")
    ax1.set_xlabel("X Grid")
    ax1.set_ylabel("Y Grid")
    ax1.set_zlabel("Probability")
    ax1.legend(loc="upper left")

    # Subplot 2: 2D Spatial Overview & Clues
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(belief_grid.T, origin='lower', cmap='cividis', extent=[0, config["grid_size"], 0, config["grid_size"]])
    ax2.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, color='cyan', linestyle='--', linewidth=2, label="Agent Path")
    ax2.scatter(path[0, 0] + 0.5, path[0, 1] + 0.5, color='white', s=100, edgecolors='black', label="Start", zorder=5)
    ax2.scatter(config["eve_true_location"][0] + 0.5, config["eve_true_location"][1] + 0.5, color='magenta', s=130, marker='*', label="EVE", zorder=5)

    for ch in config["charger_locations"]:
        ax2.scatter(ch[0] + 0.5, ch[1] + 0.5, color='lime', s=70, marker='s', label="Charger" if ch == config["charger_locations"][0] else "")

    for clue in config["clues"]:
        ax2.scatter(clue["pos"][0] + 0.5, clue["pos"][1] + 0.5, color='yellow', s=60, marker='^', label="Clue" if clue == config["clues"][0] else "")

    ax2.set_title("2D Trajectory & Clue Layout")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 1.25), ncol=3)
    plt.colorbar(im, ax=ax2, shrink=0.7, label="Posterior Weight")

    # Subplot 3: Shannon Entropy Convergence
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(entropy_hist, color='orange', linewidth=2.2)
    ax3.set_title("Uncertainty Convergence (Entropy)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Shannon Entropy (bits)")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("walle_path_visualization.png", dpi=300)
    print("\n[SUCCESS] Visualizations saved as 'walle_path_visualization.png'")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_mission()