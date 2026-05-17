import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional


def safe_mkdir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def add_sumo_tools_to_path() -> Optional[Path]:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        return None
    tools_path = Path(sumo_home) / "tools"
    if tools_path.exists() and str(tools_path) not in sys.path:
        sys.path.append(str(tools_path))
    return tools_path


def detect_sumo_cmd(use_gui: bool = False) -> Optional[str]:
    binary = "sumo-gui" if use_gui else "sumo"
    found = shutil.which(binary)
    if found:
        return found

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        exe = f"{binary}.exe" if os.name == "nt" else binary
        candidate = Path(sumo_home) / "bin" / exe
        if candidate.exists():
            return str(candidate)
    return None


def detect_netgenerate_cmd() -> Optional[str]:
    found = shutil.which("netgenerate")
    if found:
        return found
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        exe = "netgenerate.exe" if os.name == "nt" else "netgenerate"
        candidate = Path(sumo_home) / "bin" / exe
        if candidate.exists():
            return str(candidate)
    return None


def find_scenario_files(scenario_dir: Path) -> Dict[str, Optional[Path]]:
    scenario_dir = Path(scenario_dir)
    try:
        from .scenario import discover_scenario

        scenario = discover_scenario(scenario_dir, end=300.0, step_length=1.0)
        if scenario.found:
            return scenario.as_legacy_dict()
    except Exception:
        pass

    def first_existing(preferred: str, pattern: str) -> Optional[Path]:
        preferred_path = scenario_dir / preferred
        if preferred_path.exists():
            return preferred_path
        matches = sorted(scenario_dir.glob(pattern))
        return matches[0] if matches else None

    return {
        "sumocfg": None,
        "net": first_existing("net.xml", "*.net.xml"),
        "rou": first_existing("rou.xml", "*.rou.xml"),
        "add": first_existing("add.xml", "*.add.xml"),
        "route_files": [],
        "additional_files": [],
        "found": False,
    }


def check_sumo_installation() -> tuple:
    sumo_home = os.environ.get("SUMO_HOME")
    tools_path = add_sumo_tools_to_path()
    traci_ok = False
    sumolib_ok = False
    traci_error = ""
    sumolib_error = ""
    try:
        import traci  # noqa: F401

        traci_ok = True
    except Exception as exc:
        traci_error = f"{type(exc).__name__}: {exc}"
    try:
        import sumolib  # noqa: F401

        sumolib_ok = True
    except Exception as exc:
        sumolib_error = f"{type(exc).__name__}: {exc}"

    sumo_cmd = detect_sumo_cmd(use_gui=False)
    if sumo_home and tools_path and traci_ok and sumolib_ok and sumo_cmd:
        return True, ""

    details = []
    if not sumo_home:
        details.append("- Не задана переменная окружения SUMO_HOME.")
    elif not tools_path or not tools_path.exists():
        details.append(f"- В SUMO_HOME не найдена папка tools: {tools_path}")
    if not traci_ok:
        details.append(f"- Python не может импортировать traci: {traci_error}")
    if not sumolib_ok:
        details.append(f"- Python не может импортировать sumolib: {sumolib_error}")
    if not sumo_cmd:
        details.append("- Не найдена команда sumo в PATH или SUMO_HOME/bin.")
    return False, "\n".join(details)


def sumo_install_instructions(details: str = "") -> str:
    prefix = f"{details}\n\n" if details else ""
    return (
        prefix
        + "SUMO не найден или Python не видит TraCI.\n\n"
        + "Что сделать:\n"
        + "1. Установите SUMO с https://www.eclipse.org/sumo/.\n"
        + "2. Задайте переменную окружения SUMO_HOME на папку установки SUMO.\n"
        + "   Windows PowerShell, пример:\n"
        + "     [Environment]::SetEnvironmentVariable('SUMO_HOME', 'C:\\Program Files (x86)\\Eclipse\\Sumo', 'User')\n"
        + "3. Убедитесь, что в SUMO_HOME есть папки bin и tools.\n"
        + "4. Добавьте SUMO_HOME\\bin в PATH, если команда sumo не находится.\n"
        + "5. Для TraCI Python должен видеть SUMO_HOME\\tools. Этот проект добавляет tools автоматически,\n"
        + "   но SUMO_HOME должен быть задан до запуска Python.\n"
        + "6. Перезапустите терминал или PyCharm после изменения переменных окружения.\n"
    )


def check_python_dependencies() -> tuple:
    missing = []
    for module_name in ("numpy", "torch"):
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{module_name} ({type(exc).__name__}: {exc})")
    if missing:
        return False, ", ".join(missing)
    return True, ""
