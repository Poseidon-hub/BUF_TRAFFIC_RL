from pathlib import Path
from typing import Dict, List, Set
import xml.etree.ElementTree as ET


def build_tls_graph_from_net(net_xml_path: Path, debug: bool = False) -> Dict[str, List[str]]:
    graph = _build_tls_graph_with_sumolib(net_xml_path)
    if not graph:
        graph = _build_tls_graph_with_xml(net_xml_path)

    if debug:
        edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
        print(f"TLS neighbor links found: {edge_count}")
        for tls_id in sorted(graph):
            print(f"  {tls_id}: {', '.join(graph[tls_id]) if graph[tls_id] else '-'}")
    return graph


def _build_tls_graph_with_sumolib(net_xml_path: Path) -> Dict[str, List[str]]:
    try:
        from .utils import add_sumo_tools_to_path

        add_sumo_tools_to_path()
        import sumolib

        net = sumolib.net.readNet(str(net_xml_path), withInternal=True)
        traffic_lights = list(net.getTrafficLights())
    except Exception:
        return {}

    tls_ids = []
    incoming_edges: Dict[str, Set[str]] = {}
    outgoing_edges: Dict[str, Set[str]] = {}
    junctions: Dict[str, Set[str]] = {}

    for tls in traffic_lights:
        try:
            tls_id = tls.getID()
        except Exception:
            continue
        tls_ids.append(tls_id)
        incoming_edges[tls_id] = set()
        outgoing_edges[tls_id] = set()
        junctions[tls_id] = set()
        try:
            connections = tls.getConnections()
        except Exception:
            connections = []
        for connection in connections:
            try:
                in_lane = connection[0]
                out_lane = connection[1]
                in_edge = in_lane.getEdge()
                out_edge = out_lane.getEdge()
            except Exception:
                continue
            try:
                incoming_edges[tls_id].add(in_edge.getID())
                junctions[tls_id].add(in_edge.getToNode().getID())
            except Exception:
                pass
            try:
                outgoing_edges[tls_id].add(out_edge.getID())
                junctions[tls_id].add(out_edge.getFromNode().getID())
            except Exception:
                pass

    if not tls_ids:
        return {}

    graph: Dict[str, Set[str]] = {tls_id: set() for tls_id in tls_ids}

    for src in tls_ids:
        for dst in tls_ids:
            if src == dst:
                continue
            if outgoing_edges.get(src, set()).intersection(incoming_edges.get(dst, set())):
                graph[src].add(dst)
                graph[dst].add(src)

    try:
        edges = list(net.getEdges())
    except Exception:
        edges = []
    for edge in edges:
        try:
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
        except Exception:
            continue
        for src in tls_ids:
            if from_node not in junctions.get(src, set()):
                continue
            for dst in tls_ids:
                if src != dst and to_node in junctions.get(dst, set()):
                    graph[src].add(dst)
                    graph[dst].add(src)

    return {tls_id: sorted(neighbors) for tls_id, neighbors in graph.items()}


def _build_tls_graph_with_xml(net_xml_path: Path) -> Dict[str, List[str]]:
    try:
        tree = ET.parse(net_xml_path)
        root = tree.getroot()
    except Exception:
        return {}

    tls_ids: Set[str] = {node.attrib.get("id", "") for node in root.findall("tlLogic")}
    tls_ids.update(
        junction.attrib.get("id", "")
        for junction in root.findall("junction")
        if junction.attrib.get("type") == "traffic_light"
    )
    tls_ids.discard("")

    graph: Dict[str, Set[str]] = {tls_id: set() for tls_id in tls_ids}
    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue
        src = edge.attrib.get("from")
        dst = edge.attrib.get("to")
        if src in tls_ids and dst in tls_ids and src != dst:
            graph[src].add(dst)
            graph[dst].add(src)

    return {tls_id: sorted(neighbors) for tls_id, neighbors in graph.items()}


def aggregate_neighbors(tls_id: str, stats_dict: dict, neighbors_dict: dict) -> dict:
    neighbors = neighbors_dict.get(tls_id, [])
    present = [stats_dict[n] for n in neighbors if n in stats_dict]
    if not present:
        return {
            "mean_queue": 0.0,
            "mean_wait": 0.0,
            "mean_pressure": 0.0,
            "max_queue": 0.0,
        }

    mean_queue = sum(float(item.get("queue", 0.0)) for item in present) / len(present)
    mean_wait = sum(float(item.get("waiting_time", 0.0)) for item in present) / len(present)
    mean_pressure = sum(float(item.get("pressure", 0.0)) for item in present) / len(present)
    max_queue = max(float(item.get("queue", 0.0)) for item in present)
    return {
        "mean_queue": mean_queue,
        "mean_wait": mean_wait,
        "mean_pressure": mean_pressure,
        "max_queue": max_queue,
    }
