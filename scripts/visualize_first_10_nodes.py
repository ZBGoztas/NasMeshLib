"""
İlk 10 node'u görselleştir - Farklı renk, büyük boyut, degree göster
"""

import sys
from pathlib import Path
import numpy as np
import pyvista as pv

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ParserNas import ParserNas
from utils.canonical_order import apply_canonical_order
from utils.geometry_features import GeometryFeatures
from utils.logger_utils import get_context_logger


def visualize_first_10_nodes(mesh_file: str):
    """
    İlk 10 node'u bul, farklı renkte göster, büyük olarak, degree'leri yanında yaz.
    
    Args:
        mesh_file: Mesh dosyasının tam yolu
    """
    if not Path(mesh_file).exists():
        print(f"Hata: Mesh dosyası bulunamadı: {mesh_file}")
        return False
    
    file_name = Path(mesh_file).stem
    logger = get_context_logger("visualize_first_10_nodes", file_name=file_name)
    
    try:
        # Mesh'i yükle
        logger.info(f"Mesh yükleniyor: {mesh_file}")
        parser = ParserNas()
        raw_data = parser.parse_file(Path(mesh_file))
        
        # Node ve element'leri sırala
        logger.info("Node ve element'ler sıralanıyor...")
        ordered_data = apply_canonical_order(nodes=raw_data['nodes'], elements=raw_data['elements'], file_name=file_name)
        
        nodes = ordered_data['nodes']
        elements = ordered_data['elements']
        
        logger.info(f"Mesh yüklendi: {len(nodes)} node, {len(elements)} element")
        
        # Geometrik özellikler hesapla
        logger.info("Geometrik özellikler hesaplanıyor...")
        geom_extractor = GeometryFeatures(file_name=file_name)
        geometry_features = geom_extractor.extract_geometry_features(nodes, elements)
        node_degrees = geometry_features['node_degrees']
        
        logger.info(f"Node degree'ler hesaplandı")
        
        # İlk 10 node'u bul (ID'ye göre sıralanmış)
        node_ids_sorted = sorted(nodes.keys())
        first_10_nodes = node_ids_sorted[:10]
        first_10_node_degrees = {nid: node_degrees.get(nid, 0) for nid in first_10_nodes}
        
        logger.info(f"İlk 10 node: {first_10_nodes}")
        
        # PyVista görselleştir
        logger.info("PyVista penceresi açılıyor...")
        plotter = pv.Plotter(window_size=([1400, 900]))
        
        # 1. Tüm mesh'i gri olarak göster
        nid_list = sorted(nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
        points = np.array([nodes[nid] for nid in nid_list], dtype=np.float32)
        
        # Tüm mesh'in cell array'ı
        cells_array = []
        elem_ids = sorted(elements.keys())
        
        for elem_id in elem_ids:
            elem_data = elements[elem_id]
            elem_nodes = elem_data.get('nodes', [])
            
            if len(elem_nodes) >= 3:
                indices = [nid_to_idx[nid] for nid in elem_nodes if nid in nid_to_idx]
                
                if len(indices) >= 3:
                    cells_array.append(len(indices))
                    cells_array.extend(indices)
        
        cells = np.array(cells_array, dtype=np.int32)
        mesh = pv.PolyData(points, cells)
        
        # Gri mesh'i göster
        plotter.add_mesh(
            mesh,
            color=(0.85, 0.85, 0.85),
            edge_color='black',
            show_edges=True,
            opacity=0.7,
            line_width=0.5,
        )
        
        # 2. İlk 10 node'u farklı renkle ve büyük olarak göster
        first_10_coords = np.array([nodes[nid] for nid in first_10_nodes], dtype=np.float32)
        
        # Her node için ayrı rengini belirle
        colors = [
            (1.0, 0.0, 0.0),      # Kırmızı
            (0.0, 1.0, 0.0),      # Yeşil
            (0.0, 0.0, 1.0),      # Mavi
            (1.0, 1.0, 0.0),      # Sarı
            (1.0, 0.0, 1.0),      # Magenta
            (0.0, 1.0, 1.0),      # Cyan
            (1.0, 0.5, 0.0),      # Turuncu
            (0.5, 0.0, 1.0),      # Mor
            (0.0, 0.5, 1.0),      # Açık Mavi
            (1.0, 0.0, 0.5),      # Pembe
        ]
        
        # İlk 10 node'u ekleme
        for i, nid in enumerate(first_10_nodes):
            coord = nodes[nid]
            color = colors[i % len(colors)]
            
            # Büyük node (sphere)
            sphere = pv.Sphere(radius=5.0, center=coord)
            plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
            )
            
            # Node'un yanına numara yaz
            plotter.add_text(
                f"N{nid}",
                position=coord + np.array([25, 0, 0]),
                font_size=12,
                font='courier',
                color='black',
            )
        
        # 3. Sağ üst köşeye degree'leri yazıdır
        degrees_text = "İlk 10 Node'un Degree'leri:\n"
        degrees_text += "=" * 35 + "\n\n"
        
        color_names = [
            "Kırmızı", "Yeşil", "Mavi", "Sarı", "Magenta",
            "Cyan", "Turuncu", "Mor", "Açık Mavi", "Pembe"
        ]
        
        for i, nid in enumerate(first_10_nodes):
            degree = first_10_node_degrees[nid]
            color_name = color_names[i]
            degrees_text += f"Node {nid:5d} ({color_name:11s}): {degree}\n"
        
        # Metin widget'ını sağ üst köşeye ekle
        plotter.add_text(
            degrees_text,
            position=(0.98, 0.98),
            font_size=11,
            font='courier',
            color='black',
            shadow=True,
        )
        
        # Başlık ekle
        title_text = f"{file_name} | İlk 10 Node (Renkli & Büyük) | Tüm Mesh (Gri)"
        plotter.add_text(title_text, position='upper_left', font_size=13)
        
        logger.info("Görselleştirme gösteriliyor...")
        plotter.show()
        
        logger.info("Tamamlandı")
        return True
        
    except Exception as e:
        logger.error(f"Hata: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <mesh.nas>")
        sys.exit(1)
    mesh_file = sys.argv[1]

    visualize_first_10_nodes(mesh_file)