import numpy as np

class GridWorldEnv:
    def __init__(self, config):
        self.grid_size = config["grid_size"]
        self.eve_pos = np.array(config["eve_true_location"])
        self.chargers = [np.array(c) for c in config["charger_locations"]]
        self.clues = {tuple(c["pos"]): c for c in config["clues"]}
        self.noise_std = config["noise_std"]
        self.agent_pos = np.array(config["start_pos"])
        self.battery = float(config["battery_capacity"])
        self.max_battery = float(config["battery_capacity"])
        self.costs = config["action_costs"]
        self.discovered_clues = set()

    def get_signal_reading(self, pos):
        dist = np.linalg.norm(self.eve_pos - pos)
        base_signal = 100.0 / (1.0 + 0.15 * (dist ** 2))
        noisy_signal = max(0.0, float(base_signal + np.random.normal(0, self.noise_std * 10)))
        bearing = float(np.arctan2(self.eve_pos[1] - pos[1], self.eve_pos[0] - pos[0]))
        noisy_bearing = bearing + float(np.random.normal(0, self.noise_std))
        return {"signal_strength": noisy_signal, "bearing": noisy_bearing, "apparent_dist": float(dist)}

    def step(self, action, target_pos=None):
        cost = 0.0
        obs = {}

        if action == "MOVE":
            step_cost = self.costs["move"]
            if self.battery < step_cost:
                return False, {"error": "Battery depleted"}
            self.agent_pos = np.array(target_pos)
            self.battery -= step_cost
            cost = step_cost

            curr_key = tuple(self.agent_pos)
            if curr_key in self.clues and curr_key not in self.discovered_clues:
                self.discovered_clues.add(curr_key)
                obs["clue"] = self.clues[curr_key]

        elif action == "SCAN":
            scan_cost = self.costs["scan"]
            if self.battery < scan_cost:
                return False, {"error": "Battery depleted"}
            self.battery -= scan_cost
            cost = scan_cost
            obs["scan"] = self.get_signal_reading(self.agent_pos)

        elif action == "RECHARGE":
            is_at_charger = any(np.array_equal(self.agent_pos, c) for c in self.chargers)
            if is_at_charger:
                gained = min(self.costs["recharge_rate"], self.max_battery - self.battery)
                self.battery += gained
                obs["recharged"] = gained
            else:
                obs["recharged"] = 0.0

        obs["position"] = self.agent_pos.copy()
        obs["battery"] = self.battery
        obs["cost"] = cost
        return True, obs