from typing import Dict, Tuple, Set, List, Any
from collections import defaultdict
import math
from utils.logger_utils import get_context_logger


class TopologyExtractor:
    """
    Mesh'in topolojik yapısını analiz eder.
    - Element'lerden edge çıkarır (yönsüz)
    - Benzersiz edge'leri bulur ve ID verir
    - Element komşuluklarını hesaplar (aynı edge'i paylaşanlar)
    - Edge uzunluğunu hesaplar
    """
    
    def __init__(self, file_name: str = "topology_extractor"):
        """
        Topology extractor'ı başlatır.
        Logger'ı initialize eder.
        """
        self.logger = get_context_logger(name="topology_extractor", file_name=file_name)
        self.edges = {}  # edge_id -> (n1, n2)
        self.edge_to_id = {}  # (n1, n2) -> edge_id (yönsüz)
        self.edge_lengths = {}  # edge_id -> length
        self.element_edges = {}  # element_id -> [edge_ids]
        self.element_neighbors = {}  # element_id -> {neighbor_element_ids}
        self.edge_elements = {}  # edge_id -> [element_ids] (aynı edge'i paylaşan elementler)
    
    def _normalize_edge(self, n1: int, n2: int) -> Tuple[int, int]:
        """
        Edge'i yönsüz hale getir.
        (n1, n2) ve (n2, n1) aynı edge olarak kabul edilir.
        Daha küçük node ID'si önce gelecek şekilde normalize et.
        """
        return (min(n1, n2), max(n1, n2))
    
    def _extract_edges_from_elements(self, elements: Dict[int, Dict[str, Any]]) -> Dict[Tuple[int, int], List[int]]:
        """
        Element'lerden edge'leri çıkar.
        Yönsüz edge'leri benzersiz hale getir.
        
        Returns:
            edge -> [element_ids] mapping
        """
        self.logger.info(f"Element'lerden edge'ler çıkarılıyor... ({len(elements)} element)")
        
        edge_to_elements = defaultdict(list)
        
        for elem_id, elem_data in elements.items():
            nodes = elem_data.get("nodes", [])
            
            # Element'in düğümlerinden edge'ler oluştur
            # Düğümleri sırasıyla bağla (son düğüm ilk düğüme bağlanır)
            for i in range(len(nodes)):
                n1 = nodes[i]
                n2 = nodes[(i + 1) % len(nodes)]  # Döngüsel bağlantı
                
                # Edge'i yönsüz hale getir
                edge = self._normalize_edge(n1, n2)
                edge_to_elements[edge].append(elem_id)
        
        self.logger.info(f"Toplam {len(edge_to_elements)} benzersiz edge bulundu")
        return edge_to_elements
    
    def _calculate_edge_length(self, n1: int, n2: int, nodes: Dict[int, Tuple[float, float, float]]) -> float:
        """
        İki node arasındaki Euclidean uzaklığı hesapla.
        
        Args:
            n1, n2: Node ID'leri
            nodes: Node koordinatları
            
        Returns:
            Mesafe (float)
        """
        if n1 not in nodes or n2 not in nodes:
            self.logger.warning(f"Node {n1} veya {n2} bulunamadı")
            return 0.0
        
        x1, y1, z1 = nodes[n1]
        x2, y2, z2 = nodes[n2]
        
        distance = math.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
        )
        
        return distance
    
    def _build_element_edges(self, elements: Dict[int, Dict[str, Any]], edge_to_id: Dict[Tuple[int, int], int]) -> Dict[int, List[int]]:
        """
        Her element'in edge ID'lerini bul.
        
        Returns:
            element_id -> [edge_ids]
        """
        self.logger.info("Element-edge mapping oluşturuluyor...")
        
        element_edges = {}
        
        for elem_id, elem_data in elements.items():
            nodes = elem_data.get("nodes", [])
            edge_ids = []
            
            # Element'in tüm edge'lerini bul
            for i in range(len(nodes)):
                n1 = nodes[i]
                n2 = nodes[(i + 1) % len(nodes)]
                edge = self._normalize_edge(n1, n2)
                
                if edge in edge_to_id:
                    edge_ids.append(edge_to_id[edge])
            
            element_edges[elem_id] = edge_ids
        
        self.logger.info("Element-edge mapping tamamlandı")
        return element_edges
    
    def _build_element_neighbors(self, elements: Dict[int, Dict[str, Any]], element_edges: Dict[int, List[int]]) -> Dict[int, Set[int]]:
        """
        Element komşuluklarını hesapla.
        Aynı edge'i paylaşan element'ler komşu kabul edilir.
        
        Returns:
            element_id -> {neighbor_element_ids}
        """
        self.logger.info("Element komşulukları hesaplanıyor...")
        
        element_neighbors = {elem_id: set() for elem_id in elements.keys()}
        
        # Her edge için, o edge'i paylaşan element'leri bul
        edge_to_elements = defaultdict(set)
        
        for elem_id, edge_ids in element_edges.items():
            for edge_id in edge_ids:
                edge_to_elements[edge_id].add(elem_id)
        
        # Aynı edge'i paylaşan element'leri komşu ekle
        for edge_id, elem_ids in edge_to_elements.items():
            elem_list = list(elem_ids)
            # Tüm element çiftlerini komşu yap
            for i, elem1 in enumerate(elem_list):
                for elem2 in elem_list[i+1:]:
                    element_neighbors[elem1].add(elem2)
                    element_neighbors[elem2].add(elem1)
        
        self.logger.info("Element komşulukları hesaplandı")
        return element_neighbors
    
    def _build_edge_elements(self, element_edges: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Her edge'i kullanan element'leri bul.
        
        Returns:
            edge_id -> [element_ids]
        """
        self.logger.info("Edge-element mapping oluşturuluyor...")
        
        edge_elements = defaultdict(list)
        
        for elem_id, edge_ids in element_edges.items():
            for edge_id in edge_ids:
                edge_elements[edge_id].append(elem_id)
        
        self.logger.info("Edge-element mapping tamamlandı")
        return dict(edge_elements)
    
    def extract_topology(self,file_name, nodes: Dict[int, Tuple[float, float, float]], elements: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mesh'in topolojik yapısını tamamen analiz et.
        
        Args:
            nodes: Node verisi (ID -> (x, y, z))
            elements: Element verisi (ID -> {nodes: [n1, n2, ...]})
            
        Returns:
            Topoloji verisi:
            {
                "edges": {edge_id -> (n1, n2)},
                "edge_to_id": {(n1, n2) -> edge_id},
                "edge_lengths": {edge_id -> length},
                "element_edges": {element_id -> [edge_ids]},
                "element_neighbors": {element_id -> {neighbor_ids}},
                "edge_elements": {edge_id -> [element_ids]},
                "edge_count": int,
                "statistics": {...}
            }
        """
        self.logger.info("Topoloji çıkarma başlatılıyor...")
        self.logger.info(f"İşlenecek: {len(nodes)} node, {len(elements)} element")
        
        # 1. Element'lerden edge'leri çıkar
        edge_to_elements = self._extract_edges_from_elements(elements)
        
        # 2. Edge'lere ID ver (0'dan başla)
        self.logger.info("Edge'lere ID veriliyor...")
        self.edges = {}
        self.edge_to_id = {}
        
        for edge_id, (edge, _) in enumerate(edge_to_elements.items()):
            self.edges[edge_id] = edge
            self.edge_to_id[edge] = edge_id
        
        # 3. Edge uzunluklarını hesapla
        self.logger.info("Edge uzunlukları hesaplanıyor...")
        self.edge_lengths = {}
        
        for edge_id, (n1, n2) in self.edges.items():
            length = self._calculate_edge_length(n1, n2, nodes)
            self.edge_lengths[edge_id] = length
        
        # 4. Element-edge mapping oluştur
        self.element_edges = self._build_element_edges(elements, self.edge_to_id)
        
        # 5. Element komşuluklarını hesapla
        self.element_neighbors = self._build_element_neighbors(elements, self.element_edges)
        
        # 6. Edge-element mapping oluştur
        self.edge_elements = self._build_edge_elements(self.element_edges)
        
        # İstatistikler
        edge_lengths_list = list(self.edge_lengths.values())
        min_length = min(edge_lengths_list) if edge_lengths_list else 0.0
        max_length = max(edge_lengths_list) if edge_lengths_list else 0.0
        avg_length = sum(edge_lengths_list) / len(edge_lengths_list) if edge_lengths_list else 0.0
        
        stats = {
            "total_nodes": len(nodes),
            "total_elements": len(elements),
            "total_edges": len(self.edges),
            "min_edge_length": min_length,
            "max_edge_length": max_length,
            "avg_edge_length": avg_length,
            "avg_neighbor_count": sum(len(n) for n in self.element_neighbors.values()) / len(elements) if elements else 0
        }
        
        result = {
            "edges": self.edges,
            "edge_to_id": self.edge_to_id,
            "edge_lengths": self.edge_lengths,
            "element_edges": self.element_edges,
            "element_neighbors": self.element_neighbors,
            "edge_elements": self.edge_elements,
            "edge_count": len(self.edges),
            "statistics": stats
        }
        
        self.logger.info("Topoloji çıkarma tamamlandı")
        self.logger.info(f"Sonuç: {len(self.edges)} edge, {stats['avg_neighbor_count']:.2f} ort. komşu")
        
        return result
    
    def get_edge_info(self, edge_id: int, include_elements: bool = True) -> Dict[str, Any]:
        """
        Belirli bir edge hakkında bilgi al.
        
        Args:
            edge_id: Edge ID
            include_elements: Bu edge'i kullanan element'leri dahil et mi?
            
        Returns:
            Edge bilgisi
        """
        if edge_id not in self.edges:
            self.logger.warning(f"Edge {edge_id} bulunamadı")
            return {}
        
        info = {
            "edge_id": edge_id,
            "nodes": self.edges[edge_id],
            "length": self.edge_lengths.get(edge_id, 0.0)
        }
        
        if include_elements:
            info["elements"] = self.edge_elements.get(edge_id, [])
            info["element_count"] = len(info["elements"])
        
        return info
    
    def get_element_info(self, elem_id: int, include_edges: bool = True) -> Dict[str, Any]:
        """
        Belirli bir element hakkında bilgi al.
        
        Args:
            elem_id: Element ID
            include_edges: Bu element'in edge'lerini dahil et mi?
            
        Returns:
            Element bilgisi
        """
        info = {
            "element_id": elem_id,
            "neighbors": list(self.element_neighbors.get(elem_id, set())),
            "neighbor_count": len(self.element_neighbors.get(elem_id, set()))
        }
        
        if include_edges:
            info["edges"] = self.element_edges.get(elem_id, [])
            info["edge_count"] = len(info["edges"])
        
        return info
    
    def find_edge_by_nodes(self, n1: int, n2: int) -> int:
        """
        İki node'un oluşturduğu edge'i bul.
        
        Args:
            n1, n2: Node ID'leri
            
        Returns:
            Edge ID veya -1 (bulunamadı)
        """
        edge = self._normalize_edge(n1, n2)
        return self.edge_to_id.get(edge, -1)


def extract_mesh_topology(file_name, nodes: Dict[int, Tuple[float, float, float]], elements: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function: Mesh'in topolojisini analiz et.
    
    Args:
        file_name: Dosya adı (logging için)
        nodes: Node verisi
        elements: Element verisi
        
    Returns:
        Topoloji verisi
    """
    extractor = TopologyExtractor(file_name=file_name)
    return extractor.extract_topology(file_name, nodes, elements)
