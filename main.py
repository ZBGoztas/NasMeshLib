from utils.logger_utils import get_context_logger,get_logger
from utils.config_utils import load_config
from utils.ParserNas import ParserNas 
from utils.topology_extractor import TopologyExtractor
from utils.geometry_features import GeometryFeatures
from utils.mesh_summary import MeshSummary
from utils.csv_utils import CsvWrite
from utils.helper_utils import node_features_connect, edge_features_connect, elements_features_connect, mapping_features_connect
from utils.canonical_order import apply_canonical_order
import os



class PairWorkFlow():

    def __init__(self) -> None:
        self.config = load_config()
        self.raw_dir = self.config.get('raw_input_dir', 'raw_input')
        self.rafined_dir = self.config.get('rafined_input_dir', 'refined_input')
        self.logger = get_logger('main.py')
        
    def raw_main(self):
        try:
            for fileName in os.listdir(self.raw_dir):
                raw_file_path = os.path.join(self.raw_dir, fileName)
                raw_data= ParserNas().parse_file(raw_file_path)
                raw_nodes = raw_data['nodes']
                raw_elements = raw_data['elements']


                '''Dikkat topolojik veriler hesaplaniyor'''
                extractor = TopologyExtractor()
                # dict_keys(['edges', 'edge_to_id', 'edge_lengths', 'element_edges', 'element_neighbors', 'edge_elements', 'edge_count', 'statistics'])
                topology = extractor.extract_topology(fileName,raw_nodes, raw_elements)
                # edge
                raw_edges = topology['edges']
                # raw_edge_to_id = topology['edge_to_id']
                raw_edge_lengths = topology['edge_lengths']
                raw_element_edges = topology['element_edges']
                raw_element_neighbors = topology['element_neighbors']

                #edge verileri birlestiriliyor
                raw_edge_features = edge_features_connect(
                    fileName=fileName,
                    edges=raw_edges,
                    edges_lenghts=raw_edge_lengths
                )




                '''Dikkat geometrik veriler hesaplaniyor'''
                extractor = GeometryFeatures(file_name=fileName)
                # dict_keys(['element_normals', 'node_normals', 'boundary_nodes', 'node_degrees', 'mean_curvature', 'gaussian_curvature', 'statistics'])
                geometry_features = extractor.extract_geometry_features(raw_nodes, raw_elements)
                # Element
                raw_element_normals = geometry_features['element_normals']
                # Node
                raw_node_normals =geometry_features["node_normals"]
                raw_boundary_nodes = geometry_features['boundary_nodes']
                raw_node_degrees = geometry_features['node_degrees']
                raw_mean_curvature = geometry_features['mean_curvature']
                raw_gaussian_curvature = geometry_features['gaussian_curvature']
                # Node verileri birlestiriliyor
                raw_node_features = node_features_connect(
                    fileName=fileName,
                    nodes=raw_nodes,
                    boundary_nodes=raw_boundary_nodes,
                    node_normals=raw_node_normals,
                    node_degrees=raw_node_degrees,
                    mean_curvature=raw_mean_curvature,
                    gaussian_curvature=raw_gaussian_curvature,
                )

                # Element verileri birlestiriliyor
                raw_element_features = elements_features_connect(
                    fileName=fileName,
                    elements=raw_elements,
                    element_normals=raw_element_normals,
                    element_edges=raw_element_edges,
                    element_neighbors=raw_element_neighbors,
                )

                # Mapping verileri birlestiriliyor
                raw_mapping_features = mapping_features_connect(
                    fileName=fileName,
                    edge_to_id=topology['edge_to_id'],
                    edge_elements=topology['edge_elements'],
                    element_edges=raw_element_edges,
                    element_neighbors=raw_element_neighbors,
                )

                # Mesh özeti hesapla
                mesh_summary_calculator = MeshSummary(file_name=fileName)
                mesh_summary_data = mesh_summary_calculator.generate_summary(raw_nodes, raw_elements)

                '''CSV dosyalarina verileri kaydet'''
                csv_writer = CsvWrite(False)
                csv_writer.csv_nodes(fileName, raw_node_features)
                csv_writer.csv_edges(fileName, raw_edge_features)
                csv_writer.csv_elements(fileName, raw_element_features)
                csv_writer.csv_mapping(fileName, raw_mapping_features)
                


                # Summary JSON'a kaydet
                csv_writer.save_summary_json(
                    fileName=fileName,
                    topo_statistics=topology['statistics'],
                    geo_statistics=geometry_features['statistics'],
                    status=False,
                    mesh_summary=mesh_summary_data
                )
                self.rafined_main(fileName=fileName)
        except Exception as e:
            self.logger.warning(f"PairWorkFlow class'inda main fonksiyonunda raw hata | Hata: {e}")

    def rafined_main(self,fileName):
        try:
            rafined_file_path = os.path.join(self.rafined_dir, fileName)
            rafined_data= ParserNas().parse_file(rafined_file_path)


            ordered_data = apply_canonical_order(nodes=rafined_data['nodes'], elements=rafined_data['elements'],use_adjacency=True, file_name=fileName)
            rafined_nodes = ordered_data['nodes']
            rafined_elements = ordered_data['elements']

            '''Dikkat topolojik veriler hesaplaniyor'''
            extractor = TopologyExtractor()
            # dict_keys(['edges', 'edge_to_id', 'edge_lengths', 'element_edges', 'element_neighbors', 'edge_elements', 'edge_count', 'statistics'])
            topology = extractor.extract_topology(fileName,rafined_nodes, rafined_elements)
            # edge
            rafined_edges = topology['edges']
            # rafined_edge_to_id = topology['edge_to_id']
            rafined_edge_lengths = topology['edge_lengths']
            rafined_element_edges = topology['element_edges']
            rafined_element_neighbors = topology['element_neighbors']

            #edge verileri birlestiriliyor
            rafined_edge_features = edge_features_connect(
                fileName=fileName,
                edges=rafined_edges,
                edges_lenghts=rafined_edge_lengths
            )
            '''Dikkat geometrik veriler hesaplaniyor'''
            extractor = GeometryFeatures(file_name=fileName)
            # dict_keys(['element_normals', 'node_normals', 'boundary_nodes', 'node_degrees', 'mean_curvature', 'gaussian_curvature', 'statistics'])
            geometry_features = extractor.extract_geometry_features(rafined_nodes, rafined_elements)
            # Element
            rafined_element_normals = geometry_features['element_normals']
            # Node
            rafined_node_normals =geometry_features["node_normals"]
            rafined_boundary_nodes = geometry_features['boundary_nodes']
            rafined_node_degrees = geometry_features['node_degrees']
            rafined_mean_curvature = geometry_features['mean_curvature']
            rafined_gaussian_curvature = geometry_features['gaussian_curvature']
            # Node verileri birlestiriliyor
            rafined_node_features = node_features_connect(
                fileName=fileName,
                nodes=rafined_nodes,
                boundary_nodes=rafined_boundary_nodes,
                node_normals=rafined_node_normals,
                node_degrees=rafined_node_degrees,
                mean_curvature=rafined_mean_curvature,
                gaussian_curvature=rafined_gaussian_curvature,
            )

            # Element verileri birlestiriliyor
            rafined_element_features = elements_features_connect(
                fileName=fileName,
                elements=rafined_elements,
                element_normals=rafined_element_normals,
                element_edges=rafined_element_edges,
                element_neighbors=rafined_element_neighbors,
            )

            # Mapping verileri birlestiriliyor
            rafined_mapping_features = mapping_features_connect(
                fileName=fileName,
                edge_to_id=topology['edge_to_id'],
                edge_elements=topology['edge_elements'],
                element_edges=rafined_element_edges,
                element_neighbors=rafined_element_neighbors,
            )

            # Mesh özeti hesapla
            mesh_summary_calculator = MeshSummary(file_name=fileName)
            mesh_summary_data = mesh_summary_calculator.generate_summary(rafined_nodes, rafined_elements)

            '''CSV dosyalarina verileri kaydet'''
            csv_writer = CsvWrite(True)
            csv_writer.csv_nodes(fileName, rafined_node_features)
            csv_writer.csv_edges(fileName, rafined_edge_features)
            csv_writer.csv_elements(fileName, rafined_element_features)
            csv_writer.csv_mapping(fileName, rafined_mapping_features)
            


            # Summary JSON'a kaydet
            csv_writer.save_summary_json(
                fileName=fileName,
                topo_statistics=topology['statistics'],
                geo_statistics=geometry_features['statistics'],
                status=True,
                mesh_summary=mesh_summary_data
            )
        except Exception as e:
            self.logger.warning(f"PairWorkFlow class'inda main fonksiyonunda rafined hata | Hata: {e}")


if __name__ == "__main__":

    app = PairWorkFlow()
    app.raw_main()