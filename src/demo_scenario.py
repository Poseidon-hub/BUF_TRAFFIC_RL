import subprocess
from pathlib import Path
from typing import Optional

from .utils import detect_netgenerate_cmd, detect_sumo_cmd, safe_mkdir


def _write_routes(route_path: Path, end_time: int = 3600) -> None:
    route_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.9" guiShape="passenger"/>

    <route id="north_south" edges="top0A0 A0bottom0"/>
    <route id="south_north" edges="bottom0A0 A0top0"/>
    <route id="west_east" edges="left0A0 A0right0"/>
    <route id="east_west" edges="right0A0 A0left0"/>

    <flow id="flow_ns" type="car" route="north_south" begin="0" end="{end_time}" period="7" departLane="best" departSpeed="max"/>
    <flow id="flow_sn" type="car" route="south_north" begin="0" end="{end_time}" period="7" departLane="best" departSpeed="max"/>
    <flow id="flow_we" type="car" route="west_east" begin="0" end="{end_time}" period="7" departLane="best" departSpeed="max"/>
    <flow id="flow_ew" type="car" route="east_west" begin="0" end="{end_time}" period="7" departLane="best" departSpeed="max"/>
</routes>
"""
    route_path.write_text(route_xml, encoding="utf-8")


def create_demo_scenario(scenario_dir: Path) -> None:
    scenario_dir = Path(scenario_dir)
    safe_mkdir(scenario_dir)
    net_path = scenario_dir / "net.xml"
    rou_path = scenario_dir / "rou.xml"

    netgenerate_cmd = detect_netgenerate_cmd()
    if not netgenerate_cmd:
        raise RuntimeError(
            "Не найдена команда netgenerate. Для автоматического demo-сценария нужен SUMO_HOME/bin/netgenerate."
        )

    cmd = [
        netgenerate_cmd,
        "--grid",
        "--grid.number",
        "1",
        "--grid.length",
        "200",
        "--grid.attach-length",
        "200",
        "--tls.set",
        "A0",
        "--tls.green.time",
        "25",
        "--tls.yellow.time",
        "3",
        "--output-file",
        str(net_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "netgenerate не смог создать demo net.xml:\n"
            + (result.stderr.strip() or result.stdout.strip() or "нет вывода")
        )

    _write_routes(rou_path)

    if not validate_demo(scenario_dir):
        raise RuntimeError("Demo-сценарий создан, но SUMO не смог запустить его на 10 шагов.")


def validate_demo(scenario_dir: Path) -> bool:
    scenario_dir = Path(scenario_dir)
    net_path = scenario_dir / "net.xml"
    rou_path = scenario_dir / "rou.xml"
    if not net_path.exists() or not rou_path.exists():
        return False

    sumo_cmd: Optional[str] = detect_sumo_cmd(use_gui=False)
    if not sumo_cmd:
        return False

    cmd = [
        sumo_cmd,
        "-n",
        str(net_path),
        "-r",
        str(rou_path),
        "--begin",
        "0",
        "--end",
        "10",
        "--step-length",
        "1",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
        "--no-warnings",
        "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
