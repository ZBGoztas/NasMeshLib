from utils.logger_utils import get_context_logger
from typing import Dict, Tuple, Set, Any
import numpy as np


def node_features_connect(
    fileName: str,
    nodes: Dict[int, Tuple[float, float, float]],
    node_normals: Dict[int, Tuple[float, float, float]],
    boundary_nodes: Set[int],
    node_degrees: Dict[int, int],
    mean_curvature: Dict[int, float],
    gaussian_curvature: Dict[int, float],
) -> Dict[int, dict]:
    """
    Tüm node feature'larını tek bir sözlükte birleştirir.
    """
    logger = get_context_logger("node_features_connect", file_name=fileName)
    features: Dict[int, dict] = {}

    try:
        common_ids = nodes.keys() & node_normals.keys()

        for nid in common_ids:
            features[nid] = {
                "coord": nodes[nid],
                "normal": node_normals[nid],
                "is_boundary": 1 if nid in boundary_nodes else 0,
                "degree": node_degrees.get(nid, 0),
                "mean_curvature": float(mean_curvature.get(nid, 0.0)),
                "gaussian_curvature": float(gaussian_curvature.get(nid, 0.0)),
            }

        logger.info(f"{fileName} | {len(features)} node için feature birleştirildi.")
        return features

    except Exception as e:
        logger.warning(f"{fileName} | node_features_connect içinde hata: {e}")
        return {}
    
def edge_features_connect(
        
        fileName:str,
        edges: dict[int, tuple[int,int]],
        edges_lenghts:  dict[int, float],
        ) -> Dict[int, dict]:
    """
    Tüm edge feature'larını tek bir sözlükte birleştirir.
    """
    logger = get_context_logger("edge_features_connect", file_name=fileName)
    features: Dict[int, dict] = {}
    try:
        common_ids = edges.keys() & edges_lenghts.keys()

        for nid in common_ids:
            features[nid] = {
                "edges": edges[nid],
                "edges_lenghts": edges_lenghts[nid],
            }

        logger.info(f"{fileName} | {len(features)} edge için feature birleştirildi.")
        return features

    except Exception as e:
        logger.warning(f"{fileName} | edge_features_connect içinde hata: {e}")
        return {}
    

def elements_features_connect(
    fileName: str,
    elements: Dict[int, Dict[str, Any]],
    element_normals: Dict[int, np.ndarray],
    element_edges: Dict[int, list],
    element_neighbors: Dict[int, set],
) -> Dict[int, dict]:
    """
    Tüm element feature'larını tek bir sözlükte birleştirir.
    
    Çıktı:
        {
            element_id: {
                "nodes": [n1, n2, n3, n4, ...],
                "normal": [nx, ny, nz],
                "edges": [e1, e2, e3, e4, ...],
                "neighbor_elements": {elem1, elem2, ...},
                "node_count": 4,
            },
            ...
        }
    """
    logger = get_context_logger("elements_features_connect", file_name=fileName)
    features: Dict[int, dict] = {}

    try:
        for elem_id, elem_data in elements.items():
            nodes_list = elem_data.get("nodes", [])
            normal = element_normals.get(elem_id, np.array([0.0, 0.0, 1.0]))
            edges = element_edges.get(elem_id, [])
            neighbors = element_neighbors.get(elem_id, set())

            features[elem_id] = {
                "nodes": nodes_list,
                "normal": tuple(normal),
                "edges": edges,
                "neighbor_elements": neighbors,
                "node_count": len(nodes_list),
            }

        logger.info(f"{fileName} | {len(features)} element için feature birleştirildi.")
        return features

    except Exception as e:
        logger.warning(f"{fileName} | elements_features_connect içinde hata: {e}")
        return {}


def mapping_features_connect(
    fileName: str,
    edge_to_id: Dict[Tuple[int, int], int],
    edge_elements: Dict[int, list],
    element_edges: Dict[int, list],
    element_neighbors: Dict[int, set],
) -> Dict[str, Any]:
    """
    Tüm mapping verilerini tek bir sözlükte birleştirir.
    
    Çıktı:
        {
            "edge_to_id": {(n1, n2): edge_id, ...},
            "edge_elements": {edge_id: [elem1, elem2, ...], ...},
            "element_edges": {elem_id: [edge1, edge2, ...], ...},
            "element_neighbors": {elem_id: {neighbor1, neighbor2, ...}, ...},
        }
    """
    logger = get_context_logger("mapping_features_connect", file_name=fileName)
    
    try:
        mapping_data = {
            "edge_to_id": dict(edge_to_id),
            "edge_elements": dict(edge_elements),
            "element_edges": dict(element_edges),
            "element_neighbors": {eid: set(neighbors) for eid, neighbors in element_neighbors.items()},
        }
        
        logger.info(f"{fileName} | Mapping veriler birleştirildi:")
        logger.info(f"  - edge_to_id: {len(edge_to_id)} unique edge")
        logger.info(f"  - edge_elements: {len(edge_elements)} edge mapping")
        logger.info(f"  - element_edges: {len(element_edges)} element mapping")
        logger.info(f"  - element_neighbors: {len(element_neighbors)} element neighbor mapping")
        
        return mapping_data

    except Exception as e:
        logger.warning(f"{fileName} | mapping_features_connect içinde hata: {e}")
        return {}