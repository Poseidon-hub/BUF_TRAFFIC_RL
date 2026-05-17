import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TimingPhase:
    duration: float
    state: str
    minDur: Optional[float] = None
    maxDur: Optional[float] = None
    name: Optional[str] = None


@dataclass
class TimingProgram:
    tls_id: str
    programID: str
    type: str
    offset: float
    phases: List[TimingPhase]


@dataclass
class TimingProfile:
    path: Path
    programs: Dict[str, TimingProgram]

    def as_dict(self) -> dict:
        return {
            tls_id: {
                "programID": program.programID,
                "type": program.type,
                "offset": program.offset,
                "phases": [
                    {
                        "duration": phase.duration,
                        "state": phase.state,
                        "minDur": phase.minDur,
                        "maxDur": phase.maxDur,
                        "name": phase.name,
                    }
                    for phase in program.phases
                ],
            }
            for tls_id, program in self.programs.items()
        }


def load_timing_profile(path) -> TimingProfile:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xml" or path.name.lower().endswith(".add.xml"):
        return TimingProfile(path=path, programs=parse_tls_timing_file(path))
    if suffix == ".csv":
        raise NotImplementedError("CSV timing profiles are not implemented yet. Use SUMO tls.add.xml.")
    if suffix == ".json":
        raise NotImplementedError("JSON timing profiles are not implemented yet. Use SUMO tls.add.xml.")
    raise ValueError(f"Unsupported timing profile format: {path}")


def parse_tls_timing_file(path: Path) -> Dict[str, TimingProgram]:
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    programs: Dict[str, TimingProgram] = {}
    for elem in root.iter():
        if _local_name(elem.tag) != "tlLogic":
            continue
        tls_id = elem.attrib.get("id", "").strip()
        if not tls_id:
            continue
        phases = []
        for phase_elem in elem:
            if _local_name(phase_elem.tag) != "phase":
                continue
            phases.append(
                TimingPhase(
                    duration=_float_attr(phase_elem, "duration", 0.0),
                    state=phase_elem.attrib.get("state", ""),
                    minDur=_optional_float_attr(phase_elem, "minDur"),
                    maxDur=_optional_float_attr(phase_elem, "maxDur"),
                    name=phase_elem.attrib.get("name"),
                )
            )
        programs[tls_id] = TimingProgram(
            tls_id=tls_id,
            programID=elem.attrib.get("programID", ""),
            type=elem.attrib.get("type", "static"),
            offset=_float_attr(elem, "offset", 0.0),
            phases=phases,
        )
    return programs


def _float_attr(elem: ET.Element, name: str, default: float) -> float:
    try:
        return float(elem.attrib.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _optional_float_attr(elem: ET.Element, name: str) -> Optional[float]:
    if name not in elem.attrib:
        return None
    try:
        return float(elem.attrib[name])
    except (TypeError, ValueError):
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
