class FixedTimeController:
    def __init__(self, period: int = 20, min_green: int = 5):
        self.period = int(period)
        self.min_green = int(min_green)
        self._last_switch = {}

    def reset(self) -> None:
        self._last_switch.clear()

    def act(self, env) -> dict:
        actions = {}
        now = float(env.sim_time)
        for tls_id in env.tls_ids:
            last_switch = self._last_switch.get(tls_id, 0.0)
            can_switch = env.time_since_switch_by_tls.get(tls_id, 0.0) >= self.min_green
            if can_switch and now - last_switch >= self.period:
                actions[tls_id] = 1
                self._last_switch[tls_id] = now
            else:
                actions[tls_id] = 0
        return actions

