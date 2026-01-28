"""
Mesh özet bilgilerini hesapla ve göster.

Mesh'in temel özelliklerini özetler:
- Bounding box (min/max)
- Toplam yüzey alanı
- Ortalama edge uzunluğu
- Node ve element sayıları
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np

from utils.logger_utils import get_context_logger
from utils.topology_extractor import TopologyExtractor


class MeshSummary:
    """Mesh'in özet bilgilerini hesapla."""
    
    def __init__(self, file_name: str):
        """
        Initialize.
        
        Args:
            file_name: Log'lar için dosya adı
        """
        self.file_name = file_name
        self.logger = get_context_logger("mesh_summary", file_name=file_name)
        
        self.summary: Dict[str, Any] = {}
    
    def calculate_bounding_box(self, nodes: Dict[int, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """
        Bounding box hesapla (min/max).
        
        Args:
            nodes: Node ID -> (x, y, z) koordinatları
            
        Returns:
            {"min": (x_min, y_min, z_min), "max": (x_max, y_max, z_max)}
        """
        self.logger.info("Bounding box hesaplanıyor...")
        
        if not nodes:
            self.logger.warning("Node yok!")
            return {"min": (0, 0, 0), "max": (0, 0, 0)}
        
        coords = np.array(list(nodes.values()), dtype=np.float32)
        
        min_point = tuple(np.min(coords, axis=0))
        max_point = tuple(np.max(coords, axis=0))
        
        self.logger.info(f"  Min: {min_point}")
        self.logger.info(f"  Max: {max_point}")
        
        return {"min": min_point, "max": max_point}
    
    def calculate_surface_area(self, nodes: Dict[int, Tuple[float, float, float]], 
                              elements: Dict[int, Dict[str, Any]]) -> float:
        """
        Toplam yüzey alanını hesapla.
        
        Her element'in alanını hesapla ve topla.
        
        Args:
            nodes: Node ID -> (x, y, z) koordinatları
            elements: Element ID -> {nodes: [node_ids], ...}
            
        Returns:
            Toplam yüzey alanı
        """
        self.logger.info("Toplam yüzey alanı hesaplanıyor...")
        
        if not elements:
            self.logger.warning("Element yok!")
            return 0.0
        
        total_area = 0.0
        
        for elem_id, elem_data in elements.items():
            node_ids = elem_data.get('nodes', [])
            
            if len(node_ids) < 3:
                continue
            
            # Element'in koordinatlarını al
            elem_coords = np.array([nodes[nid] for nid in node_ids if nid in nodes], dtype=np.float32)
            
            if len(elem_coords) < 3:
                continue
            
            # Polygon alanını hesapla (shoelace formula)
            # 3D için: centroid'e göre triangles'a böl
            area = self._calculate_polygon_area(elem_coords)
            total_area += area
        
        self.logger.info(f"  Toplam alan: {total_area:.6f}")
        return total_area
    
    def _calculate_polygon_area(self, coords: np.ndarray) -> float:
        """
        3D polygon'un alanını hesapla.
        
        Centroid'e göre triangles'a böl ve alanları topla.
        
        Args:
            coords: (n, 3) şekli numpy array - polygon'un koordinatları
            
        Returns:
            Polygon alanı
        """
        if len(coords) < 3:
            return 0.0
        
        # Centroid hesapla
        centroid = np.mean(coords, axis=0)
        
        total_area = 0.0
        
        # Her consecutive pair'i centroid ile triangle oluştur
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            
            # Triangle: centroid, p1, p2
            v1 = p1 - centroid
            v2 = p2 - centroid
            
            # Cross product -> area
            cross = np.cross(v1, v2)
            area_triangle = 0.5 * np.linalg.norm(cross)
            total_area += area_triangle
        
        return float(total_area)
    
    def calculate_average_edge_length(self, nodes: Dict[int, Tuple[float, float, float]], 
                                     elements: Dict[int, Dict[str, Any]]) -> float:
        """
        Ortalama edge uzunluğunu hesapla.
        
        Args:
            nodes: Node ID -> (x, y, z) koordinatları
            elements: Element ID -> {nodes: [node_ids], ...}
            
        Returns:
            Ortalama edge uzunluğu
        """
        self.logger.info("Ortalama edge uzunluğu hesaplanıyor...")
        
        if not elements:
            self.logger.warning("Element yok!")
            return 0.0
        
        extractor = TopologyExtractor(file_name=self.file_name)
        topology = extractor.extract_topology(
            file_name=self.file_name,
            nodes=nodes,
            elements=elements
        )
        
        edge_lengths = topology.get('edge_lengths', {})
        
        if not edge_lengths:
            self.logger.warning("Edge bulunamadı!")
            return 0.0
        
        lengths = list(edge_lengths.values())
        avg_length = np.mean(lengths)
        
        self.logger.info(f"  Toplam edge: {len(edge_lengths)}")
        self.logger.info(f"  Min: {np.min(lengths):.6f}")
        self.logger.info(f"  Max: {np.max(lengths):.6f}")
        self.logger.info(f"  Ortalama: {avg_length:.6f}")
        
        return float(avg_length)
    
    def generate_summary(self, nodes: Dict[int, Tuple[float, float, float]], 
                        elements: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mesh'in tamamen özetini üret.
        
        Args:
            nodes: Node ID -> (x, y, z) koordinatları
            elements: Element ID -> {nodes: [node_ids], ...}
            
        Returns:
            Özet bilgileri içeren dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("MESH ÖZETİ")
        self.logger.info("=" * 60)
        
        # Node ve element sayıları
        num_nodes = len(nodes)
        num_elements = len(elements)
        
        self.logger.info(f"\nNode sayısı: {num_nodes}")
        self.logger.info(f"Element sayısı: {num_elements}")
        
        # Bounding box
        self.logger.info(f"\nBounding box:")
        bbox = self.calculate_bounding_box(nodes)
        
        # Toplam yüzey alanı
        self.logger.info(f"\nYüzey alanı:")
        surface_area = self.calculate_surface_area(nodes, elements)
        
        # Ortalama edge uzunluğu
        self.logger.info(f"\nEdge uzunluğu:")
        avg_edge_length = self.calculate_average_edge_length(nodes, elements)
        
        self.logger.info("=" * 60)
        
        self.summary = {
            'num_nodes': num_nodes,
            'num_elements': num_elements,
            'bounding_box': bbox,
            'surface_area': surface_area,
            'average_edge_length': avg_edge_length,
        }
        
        return self.summary
    
    def get_summary(self) -> Dict[str, Any]:
        """Özetini döndür."""
        return self.summary
    
    def print_summary(self):
        """Özeti ekrana yazdır."""
        if not self.summary:
            self.logger.warning("Özet henüz hesaplanmadı!")
            return
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MESH ÖZETİ - SONUÇ")
        self.logger.info("=" * 60)
        
        self.logger.info(f"\nNode sayısı: {self.summary['num_nodes']}")
        self.logger.info(f"Element sayısı: {self.summary['num_elements']}")
        
        bbox = self.summary['bounding_box']
        self.logger.info(f"\nBounding box:")
        self.logger.info(f"  Min: ({bbox['min'][0]:.6f}, {bbox['min'][1]:.6f}, {bbox['min'][2]:.6f})")
        self.logger.info(f"  Max: ({bbox['max'][0]:.6f}, {bbox['max'][1]:.6f}, {bbox['max'][2]:.6f})")
        
        self.logger.info(f"\nToplam yüzey alanı: {self.summary['surface_area']:.6f}")
        self.logger.info(f"Ortalama edge uzunluğu: {self.summary['average_edge_length']:.6f}")
        
        self.logger.info("=" * 60 + "\n")