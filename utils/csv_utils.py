
from utils.logger_utils import get_logger, get_context_logger
from utils.config_utils import load_config
import os
from pathlib import Path
import csv
import json
from typing import Dict, Any, Optional



class CsvWrite():
    def __init__(self,status:bool) -> None:
        if status == 0:
            self.config = load_config()
            raw_csv_output = self.config.get('raw_output_csv_dir','raw_output_csv')
            self.raw_output_dir = Path(raw_csv_output)
            
            # Summary klasörü
            self.summary_dir = Path(self.config.get('summary_dir', 'summary'))
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            
            # Alt klasörleri oluştur
            self.nodes_dir = self.raw_output_dir / 'nodes'
            self.edges_dir = self.raw_output_dir / 'edges'
            self.elements_dir = self.raw_output_dir / 'elements'
            self.mapping_dir = self.raw_output_dir / 'mapping'
            
            for dir_path in [self.nodes_dir, self.edges_dir, self.elements_dir, self.mapping_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Mapping alt klasörleri
            self.edge_to_id_dir = self.mapping_dir / 'edge_to_id'
            self.edge_elements_dir = self.mapping_dir / 'edge_elements'
            self.element_edges_dir = self.mapping_dir / 'element_edges'
            self.element_neighbors_dir = self.mapping_dir / 'element_neighbors'
            
            for dir_path in [self.edge_to_id_dir, self.edge_elements_dir, 
                            self.element_edges_dir, self.element_neighbors_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.logger = get_logger('csv_utils-raw')
        else:
            self.config = load_config()
            rafined_csv_output = self.config.get('rafined_output_csv_dir','rafined_output_csv')
            self.rafined_output_dir = Path(rafined_csv_output)
            
            # Summary klasörü
            self.summary_dir = Path(self.config.get('summary_dir', 'summary'))
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            
            # Alt klasörleri oluştur
            self.nodes_dir = self.rafined_output_dir / 'nodes'
            self.edges_dir = self.rafined_output_dir / 'edges'
            self.elements_dir = self.rafined_output_dir / 'elements'
            self.mapping_dir = self.rafined_output_dir / 'mapping'
            
            for dir_path in [self.nodes_dir, self.edges_dir, self.elements_dir, self.mapping_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Mapping alt klasörleri
            self.edge_to_id_dir = self.mapping_dir / 'edge_to_id'
            self.edge_elements_dir = self.mapping_dir / 'edge_elements'
            self.element_edges_dir = self.mapping_dir / 'element_edges'
            self.element_neighbors_dir = self.mapping_dir / 'element_neighbors'
            
            for dir_path in [self.edge_to_id_dir, self.edge_elements_dir, 
                            self.element_edges_dir, self.element_neighbors_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.logger = get_logger('csv_utils-rafined')         

    def _get_csv_filename(self, fileName: str, directory: Path) -> Path:
        """
        Dosya adını .nas'ten .csv'ye dönüştür ve doğru klasöre koy.
        """
        if fileName.endswith(".nas"):
            fileName = fileName.replace(".nas", ".csv")
        return directory / fileName

    def csv_nodes(self, fileName: str, node_features: dict):
        """
        Node feature'larını CSV'ye yaz.
        
        Kolonlar: node_id, coord_x, coord_y, coord_z, normal_x, normal_y, normal_z,
                  is_boundary, degree, mean_curvature, gaussian_curvature
        """
        try:
            filePath = self._get_csv_filename(fileName, self.nodes_dir)
            
            if not node_features:
                self.logger.warning(f"{fileName} | Node feature'ları boş")
                return
            
            # İlk node'dan header oluştur
            first_node = next(iter(node_features.values()))
            fieldnames = ['node_id', 'coord_x', 'coord_y', 'coord_z', 
                         'normal_x', 'normal_y', 'normal_z',
                         'is_boundary', 'degree', 'mean_curvature', 'gaussian_curvature']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for node_id, features in node_features.items():
                    coord = features.get('coord', (0, 0, 0))
                    normal = features.get('normal', (0, 0, 0))
                    
                    row = {
                        'node_id': node_id,
                        'coord_x': coord[0],
                        'coord_y': coord[1],
                        'coord_z': coord[2],
                        'normal_x': normal[0],
                        'normal_y': normal[1],
                        'normal_z': normal[2],
                        'is_boundary': features.get('is_boundary', 0),
                        'degree': features.get('degree', 0),
                        'mean_curvature': features.get('mean_curvature', 0.0),
                        'gaussian_curvature': features.get('gaussian_curvature', 0.0),
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(node_features)} node CSV'ye kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | csv_nodes hatası: {e}")

    def csv_edges(self, fileName: str, edge_features: dict):
        """
        Edge feature'larını CSV'ye yaz.
        
        Kolonlar: edge_id, node1, node2, length
        """
        try:
            filePath = self._get_csv_filename(fileName, self.edges_dir)
            
            if not edge_features:
                self.logger.warning(f"{fileName} | Edge feature'ları boş")
                return
            
            fieldnames = ['edge_id', 'node1', 'node2', 'length']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for edge_id, features in edge_features.items():
                    edge_nodes = features.get('edges', (0, 0))
                    
                    row = {
                        'edge_id': edge_id,
                        'node1': edge_nodes[0],
                        'node2': edge_nodes[1],
                        'length': features.get('edges_lenghts', 0.0),
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(edge_features)} edge CSV'ye kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | csv_edges hatası: {e}")

    def csv_elements(self, fileName: str, element_features: dict):
        """
        Element feature'larını CSV'ye yaz.
        
        Kolonlar: element_id, node_count, nodes, normal_x, normal_y, normal_z,
                  edges, neighbor_elements
        """
        try:
            filePath = self._get_csv_filename(fileName, self.elements_dir)
            
            if not element_features:
                self.logger.warning(f"{fileName} | Element feature'ları boş")
                return
            
            fieldnames = ['element_id', 'node_count', 'nodes', 'normal_x', 'normal_y', 'normal_z',
                         'edges', 'neighbor_elements']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for elem_id, features in element_features.items():
                    nodes = features.get('nodes', [])
                    normal = features.get('normal', (0, 0, 0))
                    edges = features.get('edges', [])
                    neighbors = features.get('neighbor_elements', set())
                    
                    row = {
                        'element_id': elem_id,
                        'node_count': features.get('node_count', 0),
                        'nodes': '|'.join(str(n) for n in nodes),
                        'normal_x': normal[0],
                        'normal_y': normal[1],
                        'normal_z': normal[2],
                        'edges': '|'.join(str(e) for e in edges),
                        'neighbor_elements': '|'.join(str(n) for n in neighbors),
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(element_features)} element CSV'ye kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | csv_elements hatası: {e}")

    def csv_mapping(self, fileName: str, mapping_features: dict):
        """
        Mapping feature'larını 4 ayrı CSV dosyasına kaydet.
        
        1. edge_to_id: (node1, node2) → edge_id
        2. edge_elements: edge_id → [element_ids]
        3. element_edges: element_id → [edge_ids]
        4. element_neighbors: element_id → [neighbor_element_ids]
        """
        try:
            if not mapping_features:
                self.logger.warning(f"{fileName} | Mapping feature'ları boş")
                return
            
            # 1. edge_to_id CSV
            self._save_edge_to_id_csv(fileName, mapping_features.get('edge_to_id', {}))
            
            # 2. edge_elements CSV
            self._save_edge_elements_csv(fileName, mapping_features.get('edge_elements', {}))
            
            # 3. element_edges CSV
            self._save_element_edges_csv(fileName, mapping_features.get('element_edges', {}))
            
            # 4. element_neighbors CSV
            self._save_element_neighbors_csv(fileName, mapping_features.get('element_neighbors', {}))
            
            self.logger.info(f"{fileName} | Tüm mapping veriler CSV'ye kaydedildi")
            
        except Exception as e:
            self.logger.error(f"{fileName} | csv_mapping hatası: {e}")

    def _save_edge_to_id_csv(self, fileName: str, edge_to_id: dict):
        """
        edge_to_id mapping'ini CSV'ye kaydet.
        Kolonlar: node1, node2, edge_id
        """
        try:
            filePath = self._get_csv_filename(fileName, self.edge_to_id_dir)
            
            fieldnames = ['node1', 'node2', 'edge_id']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for (node1, node2), edge_id in edge_to_id.items():
                    row = {
                        'node1': node1,
                        'node2': node2,
                        'edge_id': edge_id,
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(edge_to_id)} edge_to_id kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | _save_edge_to_id_csv hatası: {e}")

    def _save_edge_elements_csv(self, fileName: str, edge_elements: dict):
        """
        edge_elements mapping'ini CSV'ye kaydet.
        Kolonlar: edge_id, elements (pipe-separated)
        """
        try:
            filePath = self._get_csv_filename(fileName, self.edge_elements_dir)
            
            fieldnames = ['edge_id', 'elements', 'element_count']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for edge_id, elements in edge_elements.items():
                    row = {
                        'edge_id': edge_id,
                        'elements': '|'.join(str(e) for e in elements),
                        'element_count': len(elements),
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(edge_elements)} edge_elements kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | _save_edge_elements_csv hatası: {e}")

    def _save_element_edges_csv(self, fileName: str, element_edges: dict):
        """
        element_edges mapping'ini CSV'ye kaydet.
        Kolonlar: element_id, edges (pipe-separated)
        """
        try:
            filePath = self._get_csv_filename(fileName, self.element_edges_dir)
            
            fieldnames = ['element_id', 'edges', 'edge_count']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for element_id, edges in element_edges.items():
                    row = {
                        'element_id': element_id,
                        'edges': '|'.join(str(e) for e in edges),
                        'edge_count': len(edges),
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(element_edges)} element_edges kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | _save_element_edges_csv hatası: {e}")

    def _save_element_neighbors_csv(self, fileName: str, element_neighbors: dict):
        """
        element_neighbors mapping'ini CSV'ye kaydet.
        Kolonlar: element_id, neighbors (pipe-separated)
        """
        try:
            filePath = self._get_csv_filename(fileName, self.element_neighbors_dir)
            
            fieldnames = ['element_id', 'neighbors', 'neighbor_count']
            
            with open(filePath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for element_id, neighbors in element_neighbors.items():
                    row = {
                        'element_id': element_id,
                        'neighbors': '|'.join(str(n) for n in neighbors),
                        'neighbor_count': len(neighbors),
                    }
                    writer.writerow(row)
            
            self.logger.info(f"{fileName} | {len(element_neighbors)} element_neighbors kaydedildi: {filePath}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | _save_element_neighbors_csv hatası: {e}")

    def csv_nodes_old(self, fileName: str, nodes: dict):
        """
        Eski versiyon (boş).
        """
        try:
            if fileName.endswith(".nas"):
                fileName = fileName.replace(".nas", ".csv")

            filePath = self.raw_output_dir / Path(fileName)
            with open(filePath, 'a', encoding='utf-8') as f:
                pass
            
        except Exception as e:
            print(e)

    def save_summary_json(self, fileName: str, topo_statistics: dict, geo_statistics: dict, status: bool, mesh_summary: Optional[Dict[str, Any]] = None):
        """
        Tüm mesh dosyalarının summary'lerini tek bir JSON dosyasında tutar.
        Topoloji, geometri ve mesh özeti verilerini birleştirir.
        
        Çıktı: 
            - Raw için: summary_dir/raw_summary.json
            - Refined için: summary_dir/refined_summary.json
        """
        try:
            if status == False:
                summary_file = self.summary_dir / 'raw_summary.json'
            else:
                summary_file = self.summary_dir / 'refined_summary.json'

            # Topoloji istatistikleri (doğru key isimleri kullan + numpy types'ları convert et)
            topo_stats = {
                'node_count': int(topo_statistics.get('total_nodes', 0)) if topo_statistics.get('total_nodes') else 0,
                'element_count': int(topo_statistics.get('total_elements', 0)) if topo_statistics.get('total_elements') else 0,
                'edge_count': int(topo_statistics.get('total_edges', 0)) if topo_statistics.get('total_edges') else 0,
                'min_edge_length': float(topo_statistics.get('min_edge_length', 0.0)) if topo_statistics.get('min_edge_length') is not None else None,
                'max_edge_length': float(topo_statistics.get('max_edge_length', 0.0)) if topo_statistics.get('max_edge_length') is not None else None,
                'avg_edge_length': float(topo_statistics.get('avg_edge_length', 0.0)) if topo_statistics.get('avg_edge_length') is not None else None,
                'avg_neighbor_count': float(topo_statistics.get('avg_neighbor_count', 0.0)) if topo_statistics.get('avg_neighbor_count') is not None else None,
            }
            
            # Geometri istatistikleri (doğru key isimleri kullan + numpy types'ları convert et)
            geo_stats = {
                'boundary_node_count': int(geo_statistics.get('boundary_nodes_count', 0)) if geo_statistics.get('boundary_nodes_count') else 0,
                'interior_node_count': int(geo_statistics.get('interior_nodes_count', 0)) if geo_statistics.get('interior_nodes_count') else 0,
                'boundary_percentage': float((geo_statistics.get('boundary_nodes_count', 0) / geo_statistics.get('total_nodes', 1) * 100) 
                                      if geo_statistics.get('total_nodes') else 0.0),
                'avg_node_degree': float(geo_statistics.get('avg_node_degree', 0.0)) if geo_statistics.get('avg_node_degree') is not None else None,
                'mean_curvature_avg': float(geo_statistics.get('mean_curvature_avg', 0.0)) if geo_statistics.get('mean_curvature_avg') is not None else None,
                'mean_curvature_min': float(geo_statistics.get('mean_curvature_min', 0.0)) if geo_statistics.get('mean_curvature_min') is not None else None,
                'mean_curvature_max': float(geo_statistics.get('mean_curvature_max', 0.0)) if geo_statistics.get('mean_curvature_max') is not None else None,
                'gaussian_curvature_avg': float(geo_statistics.get('gaussian_curvature_avg', 0.0)) if geo_statistics.get('gaussian_curvature_avg') is not None else None,
                'gaussian_curvature_min': float(geo_statistics.get('gaussian_curvature_min', 0.0)) if geo_statistics.get('gaussian_curvature_min') is not None else None,
                'gaussian_curvature_max': float(geo_statistics.get('gaussian_curvature_max', 0.0)) if geo_statistics.get('gaussian_curvature_max') is not None else None,
            }
            
            # Mesh özeti (MeshSummary'den)
            mesh_info = {}
            if mesh_summary:
                bbox = mesh_summary.get('bounding_box', {})
                mesh_info = {
                    'num_nodes': int(mesh_summary.get('num_nodes', 0)),
                    'num_elements': int(mesh_summary.get('num_elements', 0)),
                    'surface_area': float(mesh_summary.get('surface_area', 0.0)),
                    'average_edge_length_mesh': float(mesh_summary.get('average_edge_length', 0.0)),
                    'bounding_box': {
                        'min': [float(x) for x in bbox.get('min', [0, 0, 0])] if bbox.get('min') else None,
                        'max': [float(x) for x in bbox.get('max', [0, 0, 0])] if bbox.get('max') else None,
                    }
                }
            
            # Bu mesh'in özeti
            mesh_summary_obj = {
                'topology': topo_stats,
                'geometry': geo_stats,
                'mesh_info': mesh_info,
            }
            
            # Mevcut summary.json yükle ya da yeni oluştur
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    all_summaries = json.load(f)
            else:
                all_summaries = {'mesh_summaries': {}}
            
            # Bu dosyanın özeti ekle/güncelle
            all_summaries['mesh_summaries'][fileName] = mesh_summary_obj
            
            # Güncellenmiş JSON'u kaydet
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"{fileName} | Summary JSON güncellendi (topology + geometry + mesh_info): {summary_file}")
            
        except Exception as e:
            self.logger.error(f"{fileName} | save_summary_json hatası: {e}")

