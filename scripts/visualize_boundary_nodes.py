"""
PyVista ile mesh'in sınır node'larını görselleştir.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Set, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyvista as pv
import numpy as np
from utils.ParserNas import ParserNas
from utils.canonical_order import apply_canonical_order
from utils.geometry_features import GeometryFeatures
from utils.logger_utils import get_context_logger


class BoundaryNodeVisualizer:
    """Mesh'in sınır node'larını PyVista ile görselleştir."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_name = self.file_path.stem
        self.logger = get_context_logger("visualize_boundary", file_name=self.file_name)
        
        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        self.elements: Dict[int, Dict[str, Any]] = {}
        self.boundary_nodes: Set[int] = set()
        self.geometry_features: Optional[Dict[str, Any]] = None
    
    def load_mesh(self) -> bool:
        """Mesh dosyasını yükle."""
        try:
            self.logger.info(f"Mesh yükleniyor: {self.file_path}")
            
            parser = ParserNas()
            raw_data = parser.parse_file(self.file_path)
            
            self.logger.info("Node ve element'ler sıralanıyor...")
            ordered_data = apply_canonical_order(nodes=raw_data['nodes'], elements=raw_data['elements'], file_name=self.file_name)
            
            self.nodes = ordered_data['nodes']
            self.elements = ordered_data['elements']
            
            self.logger.info(f"Mesh yüklendi: {len(self.nodes)} node, {len(self.elements)} element")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh yükleme hatası: {e}")
            return False
    
    def extract_boundary_nodes(self, expected_degree: int = 6) -> bool:
        """Sınır node'larını çıkar."""
        try:
            self.logger.info("Geometrik özellikler hesaplanıyor...")
            
            extractor = GeometryFeatures(file_name=self.file_name)
            self.geometry_features = extractor.extract_geometry_features(
                self.nodes, 
                self.elements, 
            )
            
            self.boundary_nodes = self.geometry_features['boundary_nodes']
            
            boundary_pct = 100 * len(self.boundary_nodes) / len(self.nodes) if self.nodes else 0
            self.logger.info(f"Sınır: {len(self.boundary_nodes)} / {len(self.nodes)} ({boundary_pct:.1f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Sınır node çıkarma hatası: {e}")
            return False
    
    def create_pyvista_mesh(self) -> pv.PolyData:
        """PyVista mesh oluştur."""
        self.logger.info("PyVista mesh oluşturuluyor...")
        
        node_coords = []
        node_ids = []
        
        for node_id in sorted(self.nodes.keys()):
            node_coords.append(self.nodes[node_id])
            node_ids.append(node_id)
        
        node_coords = np.array(node_coords, dtype=np.float32)
        
        # Cell array oluştur - flatten array olarak
        cells_array = []
        for elem_id in sorted(self.elements.keys()):
            elem_data = self.elements[elem_id]
            elem_nodes = elem_data['nodes']
            
            indices = []
            for nid in elem_nodes:
                if nid in self.nodes and nid in node_ids:
                    idx = node_ids.index(nid)
                    indices.append(idx)
            
            if len(indices) >= 3:
                # PyVista format: [num_nodes, idx1, idx2, ...]
                cells_array.append(len(indices))
                cells_array.extend(indices)
        
        # NumPy array'e çevir
        if cells_array:
            cells = np.array(cells_array, dtype=np.int32)
        else:
            cells = np.array([], dtype=np.int32)
        
        mesh = pv.PolyData(node_coords, cells)
        mesh['node_id'] = np.array(node_ids, dtype=np.int32)
        
        is_boundary = np.array(
            [1 if nid in self.boundary_nodes else 0 for nid in node_ids],
            dtype=np.int32
        )
        mesh['is_boundary'] = is_boundary
        
        self.logger.info(f"PyVista mesh: {mesh.n_points} nokta, {mesh.n_cells} element")
        return mesh
    
    def visualize(self, show_edges: bool = True):
        """Mesh'i görselleştir."""
        self.logger.info("Görselleştirme başlatılıyor...")
        
        mesh = self.create_pyvista_mesh()
        plotter = pv.Plotter()
        
        plotter.add_mesh(
            mesh,
            color='lightgray',
            edge_color='black' if show_edges else None,
            show_edges=show_edges,
            opacity=0.7,
            label='Mesh'
        )
        
        # Sınır node'ları kırmızı
        if len(self.boundary_nodes) > 0:
            boundary_coords = []
            for node_id in self.boundary_nodes:
                if node_id in self.nodes:
                    boundary_coords.append(self.nodes[node_id])
            
            if boundary_coords:
                boundary_coords_arr = np.array(boundary_coords, dtype=np.float32)
                boundary_cloud = pv.PolyData(boundary_coords_arr)
                
                plotter.add_mesh(
                    boundary_cloud,
                    color='red',
                    point_size=8,
                    label=f'Boundary ({len(self.boundary_nodes)})',
                    render_points_as_spheres=True
                )
        
        # İç node'ları mavi
        interior_nodes = set(self.nodes.keys()) - self.boundary_nodes
        if 0 < len(interior_nodes) < 1000:
            interior_coords = []
            for node_id in interior_nodes:
                interior_coords.append(self.nodes[node_id])
            
            if interior_coords:
                interior_coords_arr = np.array(interior_coords, dtype=np.float32)
                interior_cloud = pv.PolyData(interior_coords_arr)
                
                plotter.add_mesh(
                    interior_cloud,
                    color='blue',
                    point_size=3,
                    label=f'Interior ({len(interior_nodes)})',
                    render_points_as_spheres=True
                )
        
        # Legend göstermek istersek: label parametresi zaten eklendi
        title_text = f'{self.file_name} | Nodes: {len(self.nodes)} | Elements: {len(self.elements)}'
        plotter.add_text(title_text, position='upper_left', font_size=10)
        
        if self.geometry_features:
            stats = self.geometry_features['statistics']
            info_text = (
                f"Boundary: {stats['boundary_nodes_count']} | "
                f"Interior: {stats['interior_nodes_count']} | "
                f"Mean Curv: {stats['mean_curvature_avg']:.6f}"
            )
            plotter.add_text(info_text, position='lower_left', font_size=10)
        
        self.logger.info("PyVista penceresi açılıyor...")
        plotter.show()
        self.logger.info("Tamamlandı")


def visualize_boundary_nodes(file_path: str, expected_degree: int = 6, show_edges: bool = True) -> bool:
    """
    Mesh'in sınır node'larını görselleştir.
    
    Args:
        file_path: Mesh dosyasının tam yolu
        expected_degree: Sınır belirleme için beklenen minimum degree
        show_edges: Edge'leri göster mi?
        
    Returns:
        Başarılı mı?
    """
    visualizer = BoundaryNodeVisualizer(file_path)
    
    if not visualizer.load_mesh():
        return False
    
    if not visualizer.extract_boundary_nodes(expected_degree=expected_degree):
        return False
    
    visualizer.visualize(show_edges=show_edges)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <mesh.nas>")
        sys.exit(1)
    mesh_file = sys.argv[1]
    
    visualize_boundary_nodes(mesh_file, expected_degree=6, show_edges=True)