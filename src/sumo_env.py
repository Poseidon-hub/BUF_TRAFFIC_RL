import os
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import training_params as params
from .config import ACTION_SIZE
from .features import build_observation, encode_phase_onehot
from .graph_utils import aggregate_neighbors, build_tls_graph_from_net
from .scenario import build_sumo_command, discover_scenario, normalize_sumo_mode
from .timing_profiles import load_timing_profile
from .utils import add_sumo_tools_to_path


PEDESTRIAN_ONLY_CLASSES = {"pedestrian"}
VEHICLE_CLASSES = {
    "passenger",
    "private",
    "taxi",
    "bus",
    "coach",
    "delivery",
    "truck",
    "trailer",
    "motorcycle",
    "moped",
    "bicycle",
    "evehicle",
    "emergency",
    "authority",
    "army",
    "vip",
    "hov",
    "custom1",
    "custom2",
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


class SumoMultiAgentEnv:
    def __init__(
        self,
        scenario_dir,
        use_gui: bool = False,
        step_length: float = params.STEP_LENGTH,
        episode_seconds: int = params.EPISODE_SECONDS,
        min_green: int = params.MIN_GREEN,
        alpha: float = params.REWARD_ALPHA_QUEUE,
        beta: float = params.REWARD_BETA_NEIGHBOR,
        obs_cfg=None,
        seed: int = params.SEED,
        sumo_extra_args: Optional[List[str]] = None,
        mode: str = "rl",
    ):
        self.scenario_dir = Path(scenario_dir)
        self.use_gui = bool(use_gui)
        self.step_length = float(step_length)
        self.episode_seconds = int(episode_seconds)
        self.min_green = int(min_green)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.obs_cfg = obs_cfg
        self.seed = int(seed)
        self.sumo_extra_args = list(sumo_extra_args) if sumo_extra_args is not None else None
        self.mode = normalize_sumo_mode(mode)

        self.conn = None
        self.label = f"sumo_rl_{uuid.uuid4().hex[:8]}"
        self.traci = None
        self.scenario = None

        self.tls_ids: List[str] = []
        self.neighbors: Dict[str, List[str]] = {}
        self.num_phases_per_tls: Dict[str, int] = {}
        self.incoming_lanes: Dict[str, List[str]] = {}
        self.time_since_switch_by_tls: Dict[str, float] = {}
        self.last_phase_by_tls: Dict[str, int] = {}
        self.max_phases_global: int = 1
        self.obs_size: int = 0
        self.action_size: int = ACTION_SIZE

        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.total_arrived: int = 0
        self.total_departed: int = 0
        self.pedestrian_departed_total: int = 0
        self.pedestrian_arrived_total: int = 0
        self.pedestrian_running_max: int = 0
        self.pedestrian_waiting_count_sum: int = 0
        self.pedestrian_waiting_observation_sum: float = 0.0
        self.pedestrian_waiting_observation_count: int = 0
        self.pedestrian_waiting_time_available: bool = True
        self.pedestrian_waiting_time_note: str = ""
        self.episode_queue_sum: float = 0.0
        self.episode_wait_sum: float = 0.0
        self.episode_total_waiting_time: float = 0.0
        self.last_stats: Dict[str, dict] = {}
        self.phase_set_count: int = 0
        self.action_stats: Dict[str, object] = {}
        self.last_switch_actions_this_step = set()
        self.timing_profile = None
        self.timing_source: Dict[str, object] = {}
        self.program_set_count: int = 0
        self.last_reward_breakdown: Dict[str, object] = {}
        self._pedestrian_reward_warning_printed = False

    def reset(self) -> Dict[str, np.ndarray]:
        self.close()
        self._start_sumo()
        self.sim_time = 0.0
        self.step_count = 0
        self.total_arrived = 0
        self.total_departed = 0
        self.pedestrian_departed_total = 0
        self.pedestrian_arrived_total = 0
        self.pedestrian_running_max = 0
        self.pedestrian_waiting_count_sum = 0
        self.pedestrian_waiting_observation_sum = 0.0
        self.pedestrian_waiting_observation_count = 0
        self.pedestrian_waiting_time_available = True
        self.pedestrian_waiting_time_note = ""
        self.episode_queue_sum = 0.0
        self.episode_wait_sum = 0.0
        self.episode_total_waiting_time = 0.0
        self.phase_set_count = 0
        self.action_stats = self._new_action_stats()
        self.last_switch_actions_this_step = set()
        self.program_set_count = 0
        self.last_reward_breakdown = {}
        self._pedestrian_reward_warning_printed = False

        self.tls_ids = list(self.traci.trafficlight.getIDList())
        if not self.tls_ids:
            raise RuntimeError(
                "В сценарии не найдено traffic light controllers. RL evaluation невозможен."
            )

        self._apply_real_timing_programs_if_needed()
        self.num_phases_per_tls = {tls_id: self._get_num_phases(tls_id) for tls_id in self.tls_ids}
        self.max_phases_global = max(1, max(self.num_phases_per_tls.values()))
        if self.obs_cfg is not None and hasattr(self.obs_cfg, "set_observation_phases"):
            self.obs_cfg.set_observation_phases(self.max_phases_global)
        self.obs_size = 6 + self.max_phases_global

        self.incoming_lanes = {tls_id: self._get_incoming_lanes(tls_id) for tls_id in self.tls_ids}
        self.last_phase_by_tls = {tls_id: self._get_phase(tls_id) for tls_id in self.tls_ids}
        self.time_since_switch_by_tls = {tls_id: 0.0 for tls_id in self.tls_ids}

        graph = (
            build_tls_graph_from_net(
                self.scenario.net,
                debug=bool(getattr(self.obs_cfg, "debug_scenario", False)),
            )
            if self.scenario and self.scenario.net
            else {}
        )
        self.neighbors = {tls_id: [n for n in graph.get(tls_id, []) if n in self.tls_ids] for tls_id in self.tls_ids}

        self.last_stats = self._collect_all_stats()
        return self._build_observations(self.last_stats)

    def step(self, actions_dict: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, dict]:
        if self.traci is None:
            raise RuntimeError("Окружение не запущено. Сначала вызовите reset().")

        self._apply_actions(actions_dict)
        self.traci.simulationStep()
        self.step_count += 1
        self.sim_time = float(self.traci.simulation.getTime())
        self.total_arrived += int(self.traci.simulation.getArrivedNumber())
        self.total_departed += int(self.traci.simulation.getDepartedNumber())
        self.pedestrian_departed_total += self._get_person_event_count("departed")
        self.pedestrian_arrived_total += self._get_person_event_count("arrived")

        self._update_phase_timers()
        stats = self._collect_all_stats()
        pedestrian_metrics = self._collect_pedestrian_metrics()
        rewards = self._compute_rewards(stats, pedestrian_metrics)
        observations = self._build_observations(stats)

        avg_queue = float(np.mean([s["queue"] for s in stats.values()])) if stats else 0.0
        avg_wait = float(np.mean([s["waiting_time"] for s in stats.values()])) if stats else 0.0
        step_total_waiting = float(sum(float(s.get("waiting_sum", 0.0)) for s in stats.values()))
        self.episode_queue_sum += avg_queue
        self.episode_wait_sum += avg_wait
        self.episode_total_waiting_time += step_total_waiting
        vehicle_metrics = self._collect_vehicle_network_metrics()

        try:
            sumo_finished = int(self.traci.simulation.getMinExpectedNumber()) <= 0
        except Exception:
            sumo_finished = False
        done = self.sim_time >= self.episode_seconds or sumo_finished
        info = {
            "avg_queue": avg_queue,
            "avg_waiting_time": avg_wait,
            "total_waiting_time": self.episode_total_waiting_time,
            "throughput": self.total_arrived,
            "departed": self.total_departed,
            "arrived": self.total_arrived,
            "phase_set_count": self.phase_set_count,
            "program_set_count": self.program_set_count,
            "action_stats": self.get_action_stats(),
            "timing_source": dict(self.timing_source),
            "avg_speed": vehicle_metrics["avg_speed"],
            "avg_time_loss": vehicle_metrics["avg_time_loss"],
            "mean_reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
            "episode_avg_queue": self.episode_queue_sum / max(1, self.step_count),
            "episode_avg_waiting_time": self.episode_wait_sum / max(1, self.step_count),
            "sim_time": self.sim_time,
            "reward_breakdown": dict(self.last_reward_breakdown),
            "avg_vehicle_reward_component": float(
                self.last_reward_breakdown.get("avg_vehicle_reward_component", 0.0)
            ),
            "avg_pedestrian_reward_component": float(
                self.last_reward_breakdown.get("avg_pedestrian_reward_component", 0.0)
            ),
            "pedestrian_reward_share": float(
                self.last_reward_breakdown.get("pedestrian_reward_share", 0.0)
            ),
        }
        info.update(pedestrian_metrics)
        self.last_stats = stats
        return observations, rewards, done, info

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close(False)
            except TypeError:
                try:
                    self.conn.close()
                except Exception:
                    pass
            except Exception:
                pass
        self.conn = None
        self.traci = None

    def _start_sumo(self) -> None:
        add_sumo_tools_to_path()
        try:
            import traci
        except Exception as exc:
            raise RuntimeError(
                "Python не может импортировать traci. Проверьте SUMO_HOME и SUMO_HOME/tools."
            ) from exc

        self.scenario = discover_scenario(
            self.scenario_dir,
            begin=0.0,
            end=float(self.episode_seconds),
            step_length=self.step_length,
            auto_generate_sumocfg=True,
            scenario_sumocfg_file=getattr(self.obs_cfg, "scenario_sumocfg_file", None),
            scenario_net_file=getattr(self.obs_cfg, "scenario_net_file", None),
            scenario_route_file=getattr(self.obs_cfg, "scenario_route_file", None),
            auto_generate_pedestrians_if_missing=bool(
                getattr(self.obs_cfg, "auto_generate_pedestrians_if_missing", False)
            ),
            pedestrian_count=int(getattr(self.obs_cfg, "pedestrian_demand_count", 30)),
            pedestrian_begin=int(getattr(self.obs_cfg, "pedestrian_demand_begin", 0)),
            pedestrian_end=int(getattr(self.obs_cfg, "pedestrian_demand_end", self.episode_seconds)),
            pedestrian_prefix=str(getattr(self.obs_cfg, "pedestrian_demand_prefix", "auto_ped")),
            real_timing_file_name=str(getattr(self.obs_cfg, "real_timing_file", "tls.add.xml")),
            rl_use_real_timing_program_as_base=bool(
                getattr(self.obs_cfg, "rl_use_real_timing_program_as_base", True)
            ),
        )
        if not self.scenario.found or not self.scenario.sumocfg:
            raise RuntimeError("В scenario/ нужны *.sumocfg или net.xml (*.net.xml) и route-файлы (*.rou.xml/*.trips.xml).")

        extra_args = self.sumo_extra_args
        if extra_args is None:
            extra_args = [
                "--no-step-log",
                "true",
                "--quit-on-end",
                "true",
                "--time-to-teleport",
                "-1",
                "--duration-log.disable",
                "true",
                "--waiting-time-memory",
                str(max(1000, self.episode_seconds + 100)),
            ]
        cmd = build_sumo_command(
            self.scenario,
            use_gui=self.use_gui,
            seed=self.seed,
            step_length=self.step_length,
            extra_args=extra_args,
            end=float(self.episode_seconds),
            mode=self.mode,
        )
        if not self.use_gui:
            cmd.extend(["--no-warnings", "true"])

        try:
            traci.start(cmd, label=self.label)
            self.conn = traci.getConnection(self.label)
            self.traci = self.conn
        except Exception as exc:
            raise RuntimeError(f"SUMO/TraCI не смог запустить сценарий: {exc}") from exc

    def _mode_uses_real_timing(self) -> bool:
        return self.mode == "real_timing" or (
            self.mode == "rl"
            and bool(getattr(self.obs_cfg, "rl_use_real_timing_program_as_base", True))
        )

    def _apply_real_timing_programs_if_needed(self) -> None:
        path = getattr(self.scenario, "real_timing_file", None) if self.scenario else None
        source = {
            "file": str(path) if path else None,
            "mode": self.mode,
            "programs_loaded": 0,
            "tls_with_real_timing": [],
            "tls_without_real_timing": list(self.tls_ids),
            "tls_missing_in_network": [],
            "program_set_count": 0,
            "active_programs": {},
            "phase_count_check": {},
            "phase_duration_check": {},
            "used_in_this_mode": False,
        }
        if not path:
            self.timing_source = source
            return
        try:
            self.timing_profile = load_timing_profile(path)
        except Exception as exc:
            source["error"] = str(exc)
            self.timing_source = source
            return

        programs = self.timing_profile.programs
        source["programs_loaded"] = len(programs)
        tls_set = set(self.tls_ids)
        with_timing = [tls_id for tls_id in self.tls_ids if tls_id in programs]
        source["tls_with_real_timing"] = with_timing
        source["tls_without_real_timing"] = [tls_id for tls_id in self.tls_ids if tls_id not in programs]
        source["tls_missing_in_network"] = [tls_id for tls_id in programs if tls_id not in tls_set]

        if not self._mode_uses_real_timing():
            self.timing_source = source
            return

        source["used_in_this_mode"] = True
        for tls_id in with_timing:
            program = programs[tls_id]
            try:
                current_program = str(self.traci.trafficlight.getProgram(tls_id))
            except Exception:
                current_program = ""
            if program.programID and current_program != program.programID:
                try:
                    self.traci.trafficlight.setProgram(tls_id, program.programID)
                    self.program_set_count += 1
                    current_program = str(self.traci.trafficlight.getProgram(tls_id))
                except Exception as exc:
                    source.setdefault("program_set_errors", {})[tls_id] = str(exc)
            source["active_programs"][tls_id] = current_program
            phase_durations = self._active_phase_durations(tls_id)
            source["phase_count_check"][tls_id] = {
                "expected": len(program.phases),
                "actual": len(phase_durations),
                "matches": len(program.phases) == len(phase_durations),
            }
            source["phase_duration_check"][tls_id] = {
                "expected": [phase.duration for phase in program.phases],
                "actual": phase_durations,
                "matches": len(phase_durations) == len(program.phases)
                and all(
                    abs(float(a) - float(b.duration)) <= 1e-6
                    for a, b in zip(phase_durations, program.phases)
                ),
            }
        source["program_set_count"] = self.program_set_count
        self.timing_source = source

    def _active_program_logic(self, tls_id: str):
        try:
            active_program = str(self.traci.trafficlight.getProgram(tls_id))
        except Exception:
            active_program = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                logics = self.traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
            if logics:
                for logic in logics:
                    if str(getattr(logic, "programID", "")) == active_program:
                        return logic
                return logics[0]
        except Exception:
            pass
        try:
            logics = self.traci.trafficlight.getAllProgramLogics(tls_id)
            if logics:
                for logic in logics:
                    if str(getattr(logic, "programID", "")) == active_program:
                        return logic
                return logics[0]
        except Exception:
            pass
        return None

    def _active_phase_durations(self, tls_id: str) -> List[float]:
        logic = self._active_program_logic(tls_id)
        if logic is not None:
            return [float(getattr(phase, "duration", 0.0)) for phase in logic.phases]
        return []

    def _get_num_phases(self, tls_id: str) -> int:
        try:
            logic = self._active_program_logic(tls_id)
            if logic is not None:
                return max(1, len(logic.phases))
        except Exception:
            pass
        try:
            logics = self.traci.trafficlight.getAllProgramLogics(tls_id)
            if logics:
                return max(1, len(logics[0].phases))
        except Exception:
            pass
        try:
            state = self.traci.trafficlight.getRedYellowGreenState(tls_id)
            return 1 if state else 1
        except Exception:
            return 1

    def _get_phase_states(self, tls_id: str) -> List[str]:
        try:
            logic = self._active_program_logic(tls_id)
            if logic is not None:
                return [str(phase.state) for phase in logic.phases]
        except Exception:
            pass
        return []

    def _get_phase(self, tls_id: str) -> int:
        try:
            return int(self.traci.trafficlight.getPhase(tls_id))
        except Exception:
            return 0

    def _get_incoming_lanes(self, tls_id: str) -> List[str]:
        lanes = []
        try:
            lanes.extend(self.traci.trafficlight.getControlledLanes(tls_id))
        except Exception:
            pass

        if not lanes:
            try:
                controlled_links = self.traci.trafficlight.getControlledLinks(tls_id)
                for signal_links in controlled_links:
                    for link in signal_links:
                        if link and link[0]:
                            lanes.append(link[0])
            except Exception:
                pass

        result = []
        seen = set()
        for lane_id in lanes:
            if lane_id in seen:
                continue
            seen.add(lane_id)
            if self._is_vehicle_lane(lane_id):
                result.append(lane_id)
        return result

    def _is_vehicle_lane(self, lane_id: str) -> bool:
        if not lane_id or lane_id.startswith(":"):
            return False
        try:
            allowed = set(self.traci.lane.getAllowed(lane_id))
            if allowed and allowed.issubset(PEDESTRIAN_ONLY_CLASSES):
                return False
            if allowed and not allowed.intersection(VEHICLE_CLASSES):
                return False
        except Exception:
            return False

        try:
            edge_id = self.traci.lane.getEdgeID(lane_id)
            if "walkingarea" in edge_id.lower() or "crossing" in edge_id.lower():
                return False
        except Exception:
            pass

        return True

    def _new_action_stats(self) -> dict:
        return {
            "decision_count": 0,
            "hold_count": 0,
            "switch_count": 0,
            "blocked_by_min_green_count": 0,
            "phase_set_count": 0,
            "hold_phase_set_count": 0,
            "switch_phase_set_count": 0,
            "per_tls": {},
        }

    def _ensure_tls_action_stats(self, tls_id: str) -> dict:
        per_tls = self.action_stats.setdefault("per_tls", {})
        if tls_id not in per_tls:
            per_tls[tls_id] = {
                "hold": 0,
                "switch": 0,
                "blocked_by_min_green_count": 0,
                "phase_set_count": 0,
                "hold_phase_set_count": 0,
                "switch_phase_set_count": 0,
            }
        return per_tls[tls_id]

    def get_action_stats(self) -> dict:
        return {
            "decision_count": int(self.action_stats.get("decision_count", 0)),
            "hold_count": int(self.action_stats.get("hold_count", 0)),
            "switch_count": int(self.action_stats.get("switch_count", 0)),
            "blocked_by_min_green_count": int(
                self.action_stats.get("blocked_by_min_green_count", 0)
            ),
            "phase_set_count": int(self.action_stats.get("phase_set_count", 0)),
            "hold_phase_set_count": int(self.action_stats.get("hold_phase_set_count", 0)),
            "switch_phase_set_count": int(self.action_stats.get("switch_phase_set_count", 0)),
            "per_tls": {
                tls_id: dict(values)
                for tls_id, values in self.action_stats.get("per_tls", {}).items()
            },
        }

    def _apply_actions(self, actions_dict: Dict[str, int]) -> None:
        self.last_switch_actions_this_step = set()
        for tls_id in self.tls_ids:
            if tls_id not in actions_dict:
                continue
            action = int(actions_dict.get(tls_id, 0))
            tls_stats = self._ensure_tls_action_stats(tls_id)
            self.action_stats["decision_count"] += 1
            if action != 1:
                self.action_stats["hold_count"] += 1
                tls_stats["hold"] += 1
                current_phase = self._get_phase(tls_id)
                try:
                    self.traci.trafficlight.setPhase(tls_id, current_phase)
                    self.phase_set_count += 1
                    self.action_stats["phase_set_count"] += 1
                    self.action_stats["hold_phase_set_count"] = (
                        int(self.action_stats.get("hold_phase_set_count", 0)) + 1
                    )
                    tls_stats["phase_set_count"] += 1
                    tls_stats["hold_phase_set_count"] += 1
                except Exception:
                    pass
                continue
            self.action_stats["switch_count"] += 1
            tls_stats["switch"] += 1
            num_phases = self.num_phases_per_tls.get(tls_id, 1)
            if num_phases <= 1:
                continue
            if self.time_since_switch_by_tls.get(tls_id, 0.0) < self.min_green:
                self.action_stats["blocked_by_min_green_count"] += 1
                tls_stats["blocked_by_min_green_count"] += 1
                continue
            current_phase = self._get_phase(tls_id)
            next_phase = self._next_green_phase(tls_id, current_phase, num_phases)
            try:
                self.traci.trafficlight.setPhase(tls_id, next_phase)
                self.phase_set_count += 1
                self.action_stats["phase_set_count"] += 1
                self.action_stats["switch_phase_set_count"] = (
                    int(self.action_stats.get("switch_phase_set_count", 0)) + 1
                )
                tls_stats["phase_set_count"] += 1
                tls_stats["switch_phase_set_count"] += 1
                self.last_switch_actions_this_step.add(tls_id)
                self.last_phase_by_tls[tls_id] = next_phase
                self.time_since_switch_by_tls[tls_id] = 0.0
            except Exception:
                continue

    def _next_green_phase(self, tls_id: str, current_phase: int, num_phases: int) -> int:
        phase_states = self._get_phase_states(tls_id)
        if not phase_states:
            return (current_phase + 1) % max(1, num_phases)
        for offset in range(1, max(1, num_phases) + 1):
            candidate = (current_phase + offset) % max(1, num_phases)
            state = phase_states[candidate] if candidate < len(phase_states) else ""
            if self._is_green_phase_state(state):
                return candidate
        return (current_phase + 1) % max(1, num_phases)

    @staticmethod
    def _is_green_phase_state(state: str) -> bool:
        return ("g" in state.lower()) and ("y" not in state.lower())

    def _update_phase_timers(self) -> None:
        for tls_id in self.tls_ids:
            current_phase = self._get_phase(tls_id)
            last_phase = self.last_phase_by_tls.get(tls_id, current_phase)
            if current_phase != last_phase:
                self.last_phase_by_tls[tls_id] = current_phase
                self.time_since_switch_by_tls[tls_id] = 0.0
            else:
                self.time_since_switch_by_tls[tls_id] = self.time_since_switch_by_tls.get(tls_id, 0.0) + self.step_length

    def _collect_all_stats(self) -> Dict[str, dict]:
        return {tls_id: self._collect_local_stats(tls_id) for tls_id in self.tls_ids}

    def _get_person_event_count(self, event: str) -> int:
        if event == "departed":
            list_methods = ("getDepartedPersonIDList", "getDepartedPersonIDs")
            number_methods = ("getDepartedPersonNumber",)
        else:
            list_methods = ("getArrivedPersonIDList", "getArrivedPersonIDs")
            number_methods = ("getArrivedPersonNumber",)

        for method_name in list_methods:
            try:
                method = getattr(self.traci.simulation, method_name)
                return len(list(method()))
            except Exception:
                pass
        for method_name in number_methods:
            try:
                method = getattr(self.traci.simulation, method_name)
                return int(method())
            except Exception:
                pass
        return 0

    def _collect_pedestrian_metrics(self) -> dict:
        running_ids = []
        waiting_values = []

        try:
            running_ids = list(self.traci.person.getIDList())
        except Exception:
            running_ids = []
            self.pedestrian_waiting_time_available = False
            self.pedestrian_waiting_time_note = (
                "TraCI person domain is not available in this SUMO version."
            )

        self.pedestrian_running_max = max(self.pedestrian_running_max, len(running_ids))

        for person_id in running_ids:
            try:
                waiting_values.append(float(self.traci.person.getWaitingTime(person_id)))
            except Exception:
                self.pedestrian_waiting_time_available = False
                self.pedestrian_waiting_time_note = (
                    "TraCI person waiting time is not available in this SUMO version."
                )
                break

        if self.pedestrian_waiting_time_available:
            waiting_count = sum(1 for value in waiting_values if value > 0.0)
            self.pedestrian_waiting_count_sum += int(waiting_count)
            self.pedestrian_waiting_observation_sum += float(sum(waiting_values))
            self.pedestrian_waiting_observation_count += len(waiting_values)
            total_waiting = float(self.pedestrian_waiting_observation_sum)
            avg_waiting = total_waiting / max(1, self.pedestrian_waiting_observation_count)
            note = ""
        else:
            waiting_count = None
            total_waiting = None
            avg_waiting = None
            note = self.pedestrian_waiting_time_note

        return {
            "pedestrian_departed": self.pedestrian_departed_total,
            "pedestrian_arrived": self.pedestrian_arrived_total,
            "pedestrian_running": len(running_ids),
            "pedestrian_running_max": self.pedestrian_running_max,
            "pedestrian_waiting_count": waiting_count,
            "pedestrian_waiting_count_sum": (
                self.pedestrian_waiting_count_sum if self.pedestrian_waiting_time_available else None
            ),
            "pedestrian_total_waiting_time": total_waiting,
            "pedestrian_avg_waiting_time": avg_waiting,
            "pedestrian_waiting_time_available": self.pedestrian_waiting_time_available,
            "pedestrian_waiting_time_note": note,
        }

    def _collect_vehicle_network_metrics(self) -> dict:
        try:
            vehicle_ids = list(self.traci.vehicle.getIDList())
        except Exception:
            vehicle_ids = []

        speeds = []
        time_losses = []
        for veh_id in vehicle_ids:
            try:
                speeds.append(float(self.traci.vehicle.getSpeed(veh_id)))
            except Exception:
                pass
            try:
                time_losses.append(float(self.traci.vehicle.getTimeLoss(veh_id)))
            except Exception:
                pass

        return {
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "avg_time_loss": float(np.mean(time_losses)) if time_losses else 0.0,
        }

    def _collect_local_stats(self, tls_id: str) -> dict:
        lanes = self.incoming_lanes.get(tls_id, [])
        queue = 0.0
        wait_sum = 0.0
        vehicle_count = 0.0

        for lane_id in lanes:
            try:
                queue += float(self.traci.lane.getLastStepHaltingNumber(lane_id))
            except Exception:
                continue
            try:
                vehicle_ids = list(self.traci.lane.getLastStepVehicleIDs(lane_id))
            except Exception:
                vehicle_ids = []
            vehicle_count += float(len(vehicle_ids))
            for veh_id in vehicle_ids:
                try:
                    wait_sum += float(self.traci.vehicle.getWaitingTime(veh_id))
                except Exception:
                    continue

        waiting_time = wait_sum / max(1.0, vehicle_count)
        phase = self._get_phase(tls_id)
        return {
            "queue": queue,
            "waiting_time": waiting_time,
            "waiting_sum": wait_sum,
            "vehicle_count": vehicle_count,
            "current_phase_index": phase,
            "time_since_switch": self.time_since_switch_by_tls.get(tls_id, 0.0),
            "num_phases": self.num_phases_per_tls.get(tls_id, 1),
            "num_lanes": len(lanes),
        }

    def _compute_rewards(self, stats: Dict[str, dict], pedestrian_metrics: Optional[dict] = None) -> Dict[str, float]:
        rewards = {}
        queue_norm = max(1e-6, float(getattr(self.obs_cfg, "queue_norm", params.QUEUE_NORM)))
        wait_norm = max(1e-6, float(getattr(self.obs_cfg, "waiting_norm", params.WAIT_NORM)))
        vehicle_queue_weight = float(getattr(self.obs_cfg, "reward_vehicle_queue_weight", 1.0))
        vehicle_wait_weight = float(getattr(self.obs_cfg, "reward_vehicle_wait_weight", 1.0))
        neighbor_weight = float(getattr(self.obs_cfg, "reward_neighbor_weight", self.beta))
        pedestrian_penalty = self._compute_pedestrian_reward_penalty(pedestrian_metrics or {})
        per_tls_breakdown = {}
        for tls_id, local in stats.items():
            vehicle_queue_penalty = vehicle_queue_weight * (float(local["queue"]) / queue_norm)
            vehicle_wait_penalty = vehicle_wait_weight * (float(local["waiting_time"]) / wait_norm)
            neighbor_terms = []
            for neighbor_id in self.neighbors.get(tls_id, []):
                if neighbor_id in stats:
                    n = stats[neighbor_id]
                    neighbor_terms.append(
                        (float(n["queue"]) / queue_norm)
                        + (float(n["waiting_time"]) / wait_norm)
                    )
            neighbor_mean = sum(neighbor_terms) / len(neighbor_terms) if neighbor_terms else 0.0
            neighbor_penalty = neighbor_weight * neighbor_mean
            vehicle_component = vehicle_queue_penalty + vehicle_wait_penalty + neighbor_penalty
            reward = -(vehicle_component + pedestrian_penalty)
            stuck_phase_penalty = 0.0
            switch_penalty = 0.0
            if (
                bool(getattr(self.obs_cfg, "reward_stuck_phase_penalty_enabled", False))
                and float(local.get("time_since_switch", 0.0))
                >= float(getattr(self.obs_cfg, "reward_stuck_phase_after_seconds", 60))
                and (float(local.get("queue", 0.0)) > 0.0 or float(local.get("waiting_time", 0.0)) > 0.0)
            ):
                stuck_phase_penalty = float(getattr(self.obs_cfg, "reward_stuck_phase_penalty", 0.0))
                reward -= stuck_phase_penalty
            if tls_id in self.last_switch_actions_this_step:
                switch_penalty = float(getattr(self.obs_cfg, "reward_switch_penalty", 0.0))
                reward -= switch_penalty
            rewards[tls_id] = reward
            per_tls_breakdown[tls_id] = {
                "vehicle_queue_penalty": vehicle_queue_penalty,
                "vehicle_wait_penalty": vehicle_wait_penalty,
                "neighbor_penalty": neighbor_penalty,
                "vehicle_reward_component": vehicle_component,
                "pedestrian_penalty": pedestrian_penalty,
                "stuck_phase_penalty": stuck_phase_penalty,
                "switch_penalty": switch_penalty,
                "total_reward": reward,
            }
        self.last_reward_breakdown = self._aggregate_reward_breakdown(per_tls_breakdown)
        return rewards

    def _compute_pedestrian_reward_penalty(self, pedestrian_metrics: dict) -> float:
        if not bool(getattr(self.obs_cfg, "reward_use_pedestrians", False)):
            return 0.0
        if not bool(pedestrian_metrics.get("pedestrian_waiting_time_available", True)):
            return 0.0

        norm = max(1e-6, float(getattr(self.obs_cfg, "pedestrian_reward_normalization", 30.0)))
        wait_weight = float(getattr(self.obs_cfg, "reward_pedestrian_wait_weight", 0.10))
        running_weight = float(getattr(self.obs_cfg, "reward_pedestrian_running_weight", 0.02))
        blocked_weight = float(getattr(self.obs_cfg, "reward_pedestrian_blocked_weight", 0.05))

        avg_wait = _safe_float(pedestrian_metrics.get("pedestrian_avg_waiting_time"), 0.0)
        running = _safe_float(pedestrian_metrics.get("pedestrian_running"), 0.0)
        waiting_count = _safe_float(pedestrian_metrics.get("pedestrian_waiting_count"), 0.0)

        penalty = (
            wait_weight * (avg_wait / norm)
            + running_weight * (running / norm)
            + blocked_weight * (waiting_count / norm)
        )
        if not np.isfinite(penalty):
            return 0.0
        return max(0.0, float(penalty))

    def _aggregate_reward_breakdown(self, per_tls: Dict[str, dict]) -> dict:
        if not per_tls:
            return {
                "vehicle_queue_penalty": 0.0,
                "vehicle_wait_penalty": 0.0,
                "neighbor_penalty": 0.0,
                "pedestrian_penalty": 0.0,
                "total_reward": 0.0,
                "avg_vehicle_reward_component": 0.0,
                "avg_pedestrian_reward_component": 0.0,
                "pedestrian_reward_share": 0.0,
                "per_tls": {},
            }

        def mean_key(key: str) -> float:
            return float(np.mean([float(item.get(key, 0.0)) for item in per_tls.values()]))

        vehicle_component = mean_key("vehicle_reward_component")
        pedestrian_component = mean_key("pedestrian_penalty")
        denom = max(abs(vehicle_component) + abs(pedestrian_component), 1.0)
        pedestrian_share = abs(pedestrian_component) / denom if denom > 1e-12 else 0.0
        warning = ""
        if pedestrian_share > 0.3:
            warning = "WARNING: pedestrian reward component is too large; vehicle priority may be lost."
            if not self._pedestrian_reward_warning_printed:
                print(warning)
                self._pedestrian_reward_warning_printed = True
        return {
            "vehicle_queue_penalty": mean_key("vehicle_queue_penalty"),
            "vehicle_wait_penalty": mean_key("vehicle_wait_penalty"),
            "neighbor_penalty": mean_key("neighbor_penalty"),
            "pedestrian_penalty": pedestrian_component,
            "stuck_phase_penalty": mean_key("stuck_phase_penalty"),
            "switch_penalty": mean_key("switch_penalty"),
            "total_reward": mean_key("total_reward"),
            "avg_vehicle_reward_component": vehicle_component,
            "avg_pedestrian_reward_component": pedestrian_component,
            "pedestrian_reward_share": float(pedestrian_share),
            "warning": warning,
            "per_tls": per_tls,
        }

    def _build_observations(self, stats: Dict[str, dict]) -> Dict[str, np.ndarray]:
        observations = {}
        for tls_id, local in stats.items():
            aggs = aggregate_neighbors(tls_id, stats, self.neighbors)
            phase_onehot = encode_phase_onehot(
                int(local.get("current_phase_index", 0)),
                int(local.get("num_phases", 1)),
                self.max_phases_global,
            )
            observations[tls_id] = build_observation(
                tls_id=tls_id,
                local_stats=local,
                neighbor_aggs=aggs,
                phase_onehot=phase_onehot,
                time_since_switch=float(local.get("time_since_switch", 0.0)),
                obs_cfg=self.obs_cfg,
            )
        return observations
