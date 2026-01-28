from typing import Dict, Tuple, Set, List, Any, Optional
import numpy as np
import math
from collections import defaultdict
from utils.logger_utils import get_context_logger


class GeometryFeatures:
    """
    Mesh'in geometrik özelliklerini hesaplar.
    - Her element için normal vektör
    - Her node için ağırlıklı normal vektör
    - Sınır node'larını bulur (degree < beklenen)
    - Gaussian ve Mean curvature hesaplar
    """
    
    def __init__(self, file_name: str = "geometry_features"):
        """
        Geometry features calculator'ı başlatır.
        Logger'ı initialize eder.
        """
        self.logger = get_context_logger(name="geometry_features", file_name=file_name)
        self.element_normals = {}  # element_id -> normal vector (np.array)
        self.node_normals = {}  # node_id -> weighted normal vector (np.array)
        self.boundary_nodes = set()  # sınır node'ları
        self.node_degrees = {}  # node_id -> degree (kaç element'e ait)
        self.gaussian_curvature = {}  # node_id -> gaussian curvature
        self.mean_curvature = {}  # node_id -> mean curvature
        self.node_neighbors = {}  # node_id -> neighbor node ids
    
    def _calculate_element_normal(self, nodes_coords: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Element'in normal vektörünü hesapla (Newell metodu - stabil).
        Üçgen veya çokgen olabilir.
        
        Args:
            nodes_coords: Element'in node koordinatları
            
        Returns:
            Normal vektör (3D numpy array), normalize edilmiş
        """
        if len(nodes_coords) < 3:
            self.logger.warning(f"Element'te 3'ten az node var ({len(nodes_coords)})")
            return np.array([0.0, 0.0, 1.0])
        
        # Newell metodu kullan (robust)
        n = len(nodes_coords)
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        for i in range(n):
            v1 = np.array(nodes_coords[i], dtype=np.float64)
            v2 = np.array(nodes_coords[(i + 1) % n], dtype=np.float64)
            
            normal[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
            normal[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
            normal[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])
        
        # Normalize et
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        else:
            # Fallback: cross product ilk 3 nokta için
            v1 = np.array(nodes_coords[1], dtype=np.float64) - np.array(nodes_coords[0], dtype=np.float64)
            v2 = np.array(nodes_coords[2], dtype=np.float64) - np.array(nodes_coords[0], dtype=np.float64)
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])
        
        return normal
    
    def _calculate_element_area(self, nodes_coords: List[Tuple[float, float, float]]) -> float:
        """
        Element'in alanını hesapla (Shoelace formula genelleştirilmiş).
        
        Args:
            nodes_coords: Element'in node koordinatları
            
        Returns:
            Alan (float)
        """
        if len(nodes_coords) < 3:
            return 0.0
        
        # Üçgenlere böl ve alanlarını topla
        total_area = 0.0
        p0 = np.array(nodes_coords[0], dtype=np.float64)
        
        for i in range(1, len(nodes_coords) - 1):
            p1 = np.array(nodes_coords[i], dtype=np.float64)
            p2 = np.array(nodes_coords[i + 1], dtype=np.float64)
            
            v1 = p1 - p0
            v2 = p2 - p0
            cross = np.cross(v1, v2)
            area = np.linalg.norm(cross) / 2.0
            total_area += area
        
        return float(total_area)
    
    def _calculate_node_neighbors(self, elements: Dict[int, Dict[str, Any]]) -> Dict[int, Set[int]]:
        """
        Her node'un komşu node'larını bul (optimized).
        Node degree = Node'un ait olduğu unique element sayısı
        
        Returns:
            node_id -> {neighbor_node_ids}
        """
        self.logger.info("Node komşuları hesaplanıyor...")
        
        node_neighbors = defaultdict(set)
        node_to_elements = defaultdict(set)  # node -> element IDs
        
        # Element başına işlem
        for elem_id, elem_data in elements.items():
            nodes_list = elem_data.get("nodes", [])
            
            # Bu element'e ait olan node'ları kayıt et
            for nid in nodes_list:
                node_to_elements[nid].add(elem_id)
            
            # Element içindeki komşu node'ları bul (sadece topological komşu)
            # Komşu = Element tarafında yan yana olan node
            for i, n1 in enumerate(nodes_list):
                # Sonraki node (element'in topology'sinde komşu)
                n2 = nodes_list[(i + 1) % len(nodes_list)]
                node_neighbors[n1].add(n2)
                node_neighbors[n2].add(n1)
        
        # Node degree = Ait olduğu unique element sayısı
        self.node_degrees = {nid: len(node_to_elements[nid]) for nid in node_to_elements}
        
        self.logger.info(f"Node komşuları ve degree'ler hesaplandı ({len(node_to_elements)} node)")
        return dict(node_neighbors)
    
    def calculate_element_normals(self, nodes: Dict[int, Tuple[float, float, float]], 
                                   elements: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """
        Tüm element'lerin normal vektörlerini hesapla.
        
        Returns:
            element_id -> normal vector
        """
        self.logger.info(f"Element normal vektörleri hesaplanıyor... ({len(elements)} element)")
        
        self.element_normals = {}
        
        for elem_id, elem_data in elements.items():
            elem_nodes = elem_data.get("nodes", [])
            nodes_coords = [nodes[nid] for nid in elem_nodes if nid in nodes]
            
            if len(nodes_coords) >= 3:
                normal = self._calculate_element_normal(nodes_coords)
                self.element_normals[elem_id] = normal
        
        self.logger.info(f"Element normal vektörleri hesaplandı ({len(self.element_normals)})")
        return self.element_normals
    
    def calculate_node_normals(self, nodes: Dict[int, Tuple[float, float, float]], 
                               elements: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """
        Her node'un ağırlıklı normal vektörünü hesapla.
        Ağırlık, node'un ait olduğu element'lerin alanlarıyla orantılı.
        
        Returns:
            node_id -> weighted normal vector
        """
        self.logger.info(f"Node normal vektörleri hesaplanıyor... ({len(nodes)} node)")
        
        # Önce element normal'lerini hesapla
        if not self.element_normals:
            self.calculate_element_normals(nodes, elements)
        
        # Node'a ait element'leri bul
        node_elements = defaultdict(list)
        element_areas = {}
        
        for elem_id, elem_data in elements.items():
            elem_nodes = elem_data.get("nodes", [])
            nodes_coords = [nodes[nid] for nid in elem_nodes if nid in nodes]
            area = self._calculate_element_area(nodes_coords)
            element_areas[elem_id] = area
            
            for node_id in elem_nodes:
                if node_id in nodes:
                    node_elements[node_id].append(elem_id)
        
        # Ağırlıklı normal hesapla
        self.node_normals = {}
        
        for node_id in nodes.keys():
            weighted_normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            total_weight = 0.0
            
            for elem_id in node_elements.get(node_id, []):
                if elem_id in self.element_normals:
                    area = element_areas.get(elem_id, 0.0)
                    weighted_normal += self.element_normals[elem_id] * area
                    total_weight += area
            
            # Normalize et
            if total_weight > 1e-10:
                weighted_normal = weighted_normal / total_weight
                norm = np.linalg.norm(weighted_normal)
                if norm > 1e-10:
                    weighted_normal = weighted_normal / norm
            
            self.node_normals[node_id] = weighted_normal
        
        self.logger.info(f"Node normal vektörleri hesaplandı ({len(self.node_normals)})")
        return self.node_normals
    
    def find_boundary_nodes(self, nodes: Dict[int, Tuple[float, float, float]], 
                            elements: Dict[int, Dict[str, Any]]) -> Set[int]:
        """
        Sınır node'larını PyVista ile bul (optimized).
        extract_feature_edges metodunu kullanarak daha güvenilir sonuç al.
        
        Returns:
            Sınır node'larının kümesi
        """
        self.logger.info("Sınır node'ları bulunuyor (PyVista extract_feature_edges)...")
        
        try:
            import pyvista as pv
            
            # Node koordinatlarını array'e çevir + ID mapping
            nid_list = sorted(nodes.keys())
            nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}  # Hızlı lookup
            points = np.array([nodes[nid] for nid in nid_list], dtype=np.float32)
            
            # Cell array oluştur - flatten format (optimized)
            cells_array = []
            for elem_data in elements.values():
                elem_nodes = elem_data.get('nodes', [])
                
                if len(elem_nodes) >= 3:
                    # Node ID'lerini indeks'e çevir (dict lookup - O(1))
                    indices = [nid_to_idx[nid] for nid in elem_nodes if nid in nid_to_idx]
                    
                    if len(indices) >= 3:
                        cells_array.append(len(indices))
                        cells_array.extend(indices)
            
            if not cells_array:
                self.logger.warning("Cell array boş - sınır node'ları bulunamadı")
                self.boundary_nodes = set()
                return self.boundary_nodes
            
            cells = np.array(cells_array, dtype=np.int32)
            
            # PyVista mesh oluştur
            mesh = pv.PolyData(points, cells)
            
            # Feature edges çıkar (sınır kenarları)
            boundary_mesh = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False
            )
            
            # Sınır noktalarını bul - numpy broadcasting ile hızlandır
            boundary_points = boundary_mesh.points
            tolerance = 1e-6
            
            # Tüm noktaları bir seferde kontrol et (vectorized)
            self.boundary_nodes = set()
            
            # Broadcasting: points (N, 3) vs boundary_points (M, 3)
            # Her boundary point için en yakın distance'ı bul
            if len(boundary_points) > 0 and len(points) > 0:
                # Daha hızlı: KDTree kullan
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(points)
                    distances, indices = tree.query(boundary_points, k=1)
                    
                    # tolerance'dan daha yakın noktaları sınır olarak işaretle
                    boundary_indices = indices[distances < tolerance]
                    self.boundary_nodes = {nid_list[idx] for idx in boundary_indices}
                except ImportError:
                    # SciPy yoksa fallback - numpy broadcasting
                    for b_point in boundary_points:
                        dists = np.linalg.norm(points - b_point, axis=1)
                        close_indices = np.where(dists < tolerance)[0]
                        for idx in close_indices:
                            self.boundary_nodes.add(nid_list[idx])
            
            boundary_pct = 100 * len(self.boundary_nodes) / len(nodes) if nodes else 0
            self.logger.info(f"Sınır node'ları: {len(self.boundary_nodes)} / {len(nodes)} ({boundary_pct:.1f}%)")
            return self.boundary_nodes
            
        except ImportError:
            self.logger.error("PyVista yüklü değil - sınır node'ları bulunamadı")
            self.boundary_nodes = set()
            return self.boundary_nodes
        except Exception as e:
            self.logger.error(f"Sınır node bulma hatası: {e}")
            self.boundary_nodes = set()
            return self.boundary_nodes
    
    def calculate_mean_curvature(self, nodes: Dict[int, Tuple[float, float, float]], 
                                  elements: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """
        Mean curvature'ı hesapla (Discrete Differential Geometry - Optimized).
        Mean curvature = (1/2) * |normal değişim| / mesafe
        
        Vectorized approach ile tüm node'lar aynı anda işlenir.
        
        Returns:
            node_id -> mean curvature
        """
        self.logger.info("Mean curvature hesaplanıyor...")
        
        # Node komşuları hesapla
        if not self.node_neighbors:
            self.node_neighbors = self._calculate_node_neighbors(elements)
        
        # Node normal'lerini hesapla
        if not self.node_normals:
            self.calculate_node_normals(nodes, elements)
        
        self.mean_curvature = {}
        
        # Koordinatları numpy array'e çevir (vectorized işlem için)
        nid_list = list(nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
        coords_array = np.array([nodes[nid] for nid in nid_list], dtype=np.float64)
        
        # Normal'leri array'e çevir
        normals_array = np.array([
            self.node_normals.get(nid, np.array([0.0, 0.0, 1.0])) 
            for nid in nid_list
        ], dtype=np.float64)
        
        try:
            # SciPy KDTree ile komşu bulma (opsiyonel, hız için)
            from scipy.spatial import cKDTree
            
            for node_id in nodes.keys():
                if node_id not in nid_to_idx:
                    self.mean_curvature[node_id] = 0.0
                    continue
                
                neighbors = self.node_neighbors.get(node_id, set())
                
                if len(neighbors) < 1:
                    self.mean_curvature[node_id] = 0.0
                    continue
                
                idx = nid_to_idx[node_id]
                node_coord = coords_array[idx]
                node_normal = normals_array[idx]
                
                # Komşuların indislerini al
                neighbor_indices = np.array([nid_to_idx[n] for n in neighbors if n in nid_to_idx])
                
                if len(neighbor_indices) == 0:
                    self.mean_curvature[node_id] = 0.0
                    continue
                
                # Vectorized hesap
                neighbor_coords = coords_array[neighbor_indices]
                neighbor_normals = normals_array[neighbor_indices]
                
                edge_vecs = neighbor_coords - node_coord
                edge_lengths = np.linalg.norm(edge_vecs, axis=1)
                
                # Sıfırdan farklı edge'ler
                valid_edges = edge_lengths > 1e-10
                
                if not np.any(valid_edges):
                    self.mean_curvature[node_id] = 0.0
                    continue
                
                edge_lengths = edge_lengths[valid_edges]
                neighbor_normals_valid = neighbor_normals[valid_edges]
                
                # Normal farklarını hesapla
                normal_diffs = neighbor_normals_valid - node_normal
                normal_diff_norms = np.linalg.norm(normal_diffs, axis=1)
                
                # Curvature'ı hesapla
                curvatures = normal_diff_norms / edge_lengths
                
                # Weighted average
                curvature_sum = np.sum(curvatures * edge_lengths)
                weight_sum = np.sum(edge_lengths)
                
                self.mean_curvature[node_id] = (curvature_sum / weight_sum / 2.0) if weight_sum > 1e-10 else 0.0
        
        except (ImportError, Exception):
            # Fallback: NumPy ile (daha yavaş ama güvenli)
            for node_id in nodes.keys():
                if node_id not in nid_to_idx:
                    self.mean_curvature[node_id] = 0.0
                    continue
                
                neighbors = self.node_neighbors.get(node_id, set())
                
                if len(neighbors) < 1:
                    self.mean_curvature[node_id] = 0.0
                    continue
                
                idx = nid_to_idx[node_id]
                node_coord = coords_array[idx]
                node_normal = normals_array[idx]
                
                curvature_sum = 0.0
                weight_sum = 0.0
                
                for neighbor_id in neighbors:
                    if neighbor_id not in nid_to_idx:
                        continue
                    
                    neighbor_idx = nid_to_idx[neighbor_id]
                    neighbor_coord = coords_array[neighbor_idx]
                    neighbor_normal = normals_array[neighbor_idx]
                    
                    edge_vec = neighbor_coord - node_coord
                    edge_length = np.linalg.norm(edge_vec)
                    
                    if edge_length > 1e-10:
                        normal_diff = neighbor_normal - node_normal
                        curvature = np.linalg.norm(normal_diff) / edge_length
                        
                        curvature_sum += curvature * edge_length
                        weight_sum += edge_length
                
                self.mean_curvature[node_id] = (curvature_sum / weight_sum / 2.0) if weight_sum > 1e-10 else 0.0
        
        self.logger.info(f"Mean curvature hesaplandı ({len(self.mean_curvature)} node)")
        return self.mean_curvature
    
    def calculate_gaussian_curvature(self, nodes: Dict[int, Tuple[float, float, float]], 
                                     elements: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """
        Gaussian curvature'ı hesapla (Discrete Differential Geometry - Optimized).
        Gaussian curvature = (2π - angle_sum) / area (angle deficit yöntemi)
        
        Vectorized approach ile neighbour açıları verimli hesaplanır.
        
        Returns:
            node_id -> gaussian curvature
        """
        self.logger.info("Gaussian curvature hesaplanıyor...")
        
        # Node komşuları hesapla
        if not self.node_neighbors:
            self.node_neighbors = self._calculate_node_neighbors(elements)
        
        self.gaussian_curvature = {}
        
        # Koordinatları array'e çevir (vectorized işlem için)
        nid_list = list(nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
        coords_array = np.array([nodes[nid] for nid in nid_list], dtype=np.float64)
        
        for node_id in nodes.keys():
            neighbors_set = self.node_neighbors.get(node_id, set())
            neighbors = list(neighbors_set)
            
            if len(neighbors) < 2:
                self.gaussian_curvature[node_id] = 0.0
                continue
            
            if node_id not in nid_to_idx:
                self.gaussian_curvature[node_id] = 0.0
                continue
            
            idx = nid_to_idx[node_id]
            node_coord = coords_array[idx]
            
            # Vectorized: neighbor koordinatlarını hazırla
            neighbor_indices = np.array([nid_to_idx[n] for n in neighbors if n in nid_to_idx])
            
            if len(neighbor_indices) < 2:
                self.gaussian_curvature[node_id] = 0.0
                continue
            
            neighbor_coords = coords_array[neighbor_indices]
            vectors = neighbor_coords - node_coord
            
            # Tüm neighbor çiftleri arasındaki açıları hesapla
            angle_sum = 0.0
            n_neighbors = len(neighbor_indices)
            
            for i in range(n_neighbors):
                v1 = vectors[i]
                v2 = vectors[(i + 1) % n_neighbors]
                
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 1e-10 and norm_v2 > 1e-10:
                    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = math.acos(cos_angle)
                    angle_sum += angle
            
            # Gaussian curvature = angle deficit
            angle_deficit = 2 * math.pi - angle_sum
            self.gaussian_curvature[node_id] = angle_deficit
        
        self.logger.info(f"Gaussian curvature hesaplandı ({len(self.gaussian_curvature)} node)")
        return self.gaussian_curvature
    
    def extract_geometry_features(self, nodes: Dict[int, Tuple[float, float, float]], 
                                  elements: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tüm geometrik özellikleri hesapla.
        
        Args:
            nodes: Node verisi
            elements: Element verisi
            
        Returns:
            Geometrik özellikler:
            {
                "element_normals": {elem_id -> normal},
                "node_normals": {node_id -> normal},
                "boundary_nodes": {node_ids},
                "node_degrees": {node_id -> degree},
                "mean_curvature": {node_id -> curvature},
                "gaussian_curvature": {node_id -> curvature},
                "statistics": {...}
            }
        """
        self.logger.info("Geometrik özellikler çıkarılıyor...")
        self.logger.info(f"İşlenecek: {len(nodes)} node, {len(elements)} element")
        
        # 1. Element normal'leri hesapla
        self.calculate_element_normals(nodes, elements)
        
        # 2. Node normal'lerini hesapla
        self.calculate_node_normals(nodes, elements)
        
        # 3. Sınır node'larını bul
        self.find_boundary_nodes(nodes, elements)
        
        # 4. Node komşuları
        self.node_neighbors = self._calculate_node_neighbors(elements)
        
        # 5. Mean curvature
        self.calculate_mean_curvature(nodes, elements)
        
        # 6. Gaussian curvature
        self.calculate_gaussian_curvature(nodes, elements)
        
        # İstatistikler
        curvatures_mean = list(self.mean_curvature.values())
        curvatures_gauss = list(self.gaussian_curvature.values())
        
        stats = {
            "total_nodes": len(nodes),
            "total_elements": len(elements),
            "boundary_nodes_count": len(self.boundary_nodes),
            "interior_nodes_count": len(nodes) - len(self.boundary_nodes),
            "avg_node_degree": (sum(self.node_degrees.values()) / len(self.node_degrees)) if self.node_degrees else 0.0,
            "mean_curvature_min": min(curvatures_mean) if curvatures_mean else 0.0,
            "mean_curvature_max": max(curvatures_mean) if curvatures_mean else 0.0,
            "mean_curvature_avg": sum(curvatures_mean) / len(curvatures_mean) if curvatures_mean else 0.0,
            "gaussian_curvature_min": min(curvatures_gauss) if curvatures_gauss else 0.0,
            "gaussian_curvature_max": max(curvatures_gauss) if curvatures_gauss else 0.0,
            "gaussian_curvature_avg": sum(curvatures_gauss) / len(curvatures_gauss) if curvatures_gauss else 0.0,
        }
        
        result = {
            "element_normals": self.element_normals,
            "node_normals": self.node_normals,
            "boundary_nodes": self.boundary_nodes,
            "node_degrees": dict(self.node_degrees),
            "mean_curvature": self.mean_curvature,
            "gaussian_curvature": self.gaussian_curvature,
            "statistics": stats
        }
        
        self.logger.info("Geometrik özellikler çıkarması tamamlandı")
        self.logger.info(f"Sınır node'ları: {len(self.boundary_nodes)}, Mean curvature ortalaması: {stats['mean_curvature_avg']:.6f}")
        
        return result


def extract_geometry_features(file_name: str, nodes: Dict[int, Tuple[float, float, float]], 
                              elements: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function: Mesh'in geometrik özelliklerini hesapla.
    
    Args:
        file_name: Dosya adı (logging için)
        nodes: Node verisi
        elements: Element verisi
        
    Returns:
        Geometrik özellikler
    """
    extractor = GeometryFeatures(file_name=file_name)
    return extractor.extract_geometry_features(nodes, elements)