import numpy as np

class BeliefMap:
    def __init__(self, size):
        self.size = size
        self.grid = np.ones((size, size), dtype=np.float64) / (size * size)

    def update_scan(self, agent_pos, scan_data):
        strength = scan_data["signal_strength"]
        bearing = scan_data["bearing"]

        for x in range(self.size):
            for y in range(self.size):
                cell = np.array([x, y])
                dist = np.linalg.norm(cell - agent_pos)
                expected_strength = 100.0 / (1.0 + 0.15 * (dist ** 2))
                expected_bearing = np.arctan2(cell[1] - agent_pos[1], cell[0] - agent_pos[0])

                l_strength = np.exp(-0.5 * ((strength - expected_strength) / 8.0) ** 2)
                ang_diff = np.arctan2(np.sin(bearing - expected_bearing), np.cos(bearing - expected_bearing))
                l_bearing = np.exp(-0.5 * (ang_diff / 0.45) ** 2)

                self.grid[x, y] *= (l_strength * l_bearing)

        total = np.sum(self.grid)
        if total > 0:
            self.grid /= total

    def update_clue(self, clue):
        if "bearing_hint" in clue:
            target = np.array(clue["bearing_hint"])
            for x in range(self.size):
                for y in range(self.size):
                    cell = np.array([x, y])
                    dist = np.linalg.norm(cell - target)
                    self.grid[x, y] *= np.exp(-0.5 * (dist / 1.8) ** 2)
        total = np.sum(self.grid)
        if total > 0:
            self.grid /= total

    def get_entropy(self):
        p = self.grid[self.grid > 0]
        return float(-np.sum(p * np.log2(p)))

    def get_mode(self):
        idx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        return [int(idx[0]), int(idx[1])], float(self.grid[idx])


class WallEAgent:
    def __init__(self, config):
        self.size = config["grid_size"]
        self.chargers = [np.array(c) for c in config["charger_locations"]]
        self.belief = BeliefMap(self.size)

    def get_closest_charger(self, pos):
        dists = [np.linalg.norm(pos - c, ord=1) for c in self.chargers]
        idx = int(np.argmin(dists))
        return self.chargers[idx], dists[idx]

    def plan_next_action(self, current_pos, battery, step_num):
        charger, dist_to_charger = self.get_closest_charger(current_pos)
        cost_to_safety = dist_to_charger * 2.0
        safety_buffer = 8.0

        # If at a charger, continue charging until at least 90%
        if dist_to_charger == 0 and battery < 90.0:
            return "RECHARGE", current_pos, f"Battery recharging at hub ({battery:.1f}% -> 90%+)."

        # Return to safety if power drops near the reserve limit
        if battery <= (cost_to_safety + safety_buffer):
            if dist_to_charger == 0:
                return "RECHARGE", current_pos, "Battery low; recharging at docking hub."
            diff = np.sign(charger - current_pos)
            next_pos = current_pos.copy()
            if diff[0] != 0:
                next_pos[0] += diff[0]
            elif diff[1] != 0:
                next_pos[1] += diff[1]
            return "MOVE", next_pos, f"Low power margin ({battery:.1f}%). Routing to charger {charger.tolist()}."

        # Periodic sensor sweeps
        if step_num % 3 == 0:
            return "SCAN", current_pos, "Conducting omnidirectional signal triangulations."

        # Peak targeting: Move toward highest belief coordinates
        mode_coords, _ = self.belief.get_mode()
        target = np.array(mode_coords)
        diff = target - current_pos
        step = np.zeros(2, dtype=int)

        if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
            step[0] = int(np.sign(diff[0]))
        elif diff[1] != 0:
            step[1] = int(np.sign(diff[1]))

        next_pos = np.clip(current_pos + step, 0, self.size - 1)
        if np.array_equal(next_pos, current_pos):
            return "SCAN", current_pos, "Target zone reached; running pinpoint scan."

        return "MOVE", next_pos, f"Tracking strongest inferred signal towards {target.tolist()}."