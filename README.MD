# NasMeshLib - Mesh Analysis and Processing Library

**NasMeshLib** is a comprehensive Python library designed to process NASTRAN (`.nas`) mesh files, analyze their topological and geometric properties, and standardize the data (Canonical Order) for machine learning models or advanced simulations.

Developed as a core component of the MeshCrafter suite, this tool processes both raw and refined mesh data, exporting detailed features into structured CSV and JSON formats.

## 🚀 Features

*   **NASTRAN Parser:** Robustly reads `.nas` files (supporting GRID, CTRIA3, CQUAD4 cards) and fixes malformed scientific notation.
*   **Topological Analysis:**
    *   Edge extraction and length calculation.
    *   Element adjacency (neighbor) determination.
    *   Element-Edge and Edge-Element graphical mapping.
*   **Geometric Features:**
    *   Node and Element normal vector calculation.
    *   Mean Curvature and Gaussian Curvature estimation.
    *   Automatic boundary node detection.
*   **Canonical Ordering:**
    *   Spatial sorting of nodes (Z -> Y -> X).
    *   Topological re-indexing of elements using Breadth-First Search (BFS) to ensure data consistency.
*   **Visualization:** 3D rendering of meshes, boundary points, and node indices using PyVista.
*   **Data Export:** Saves all calculated features into structured CSV files and summary statistics into JSON.

## 📂 Project Structure

```text
root/
├── main.py                          # Main execution pipeline
├── scripts/                         # Visualization tools
│   ├── visualize_boundary_nodes.py  # Highlights boundary nodes
│   └── visualize_first_10_nodes.py  # Highlights specific node indices
├── utils/                           # Core library modules
│   ├── canonical_order.py           # Re-indexing algorithms
│   ├── config_utils.py              # Configuration loading
│   ├── csv_utils.py                 # CSV/JSON export handlers
│   ├── geometry_features.py         # Curvature and normal logic
│   ├── helper_utils.py              # Data unification helpers
│   ├── logger_utils.py              # Logging setup
│   ├── mesh_summary.py              # Bounding box & stats
│   ├── ParserNas.py                 # .nas file parser
│   ├── pyvista_compat.py            # PyVista compatibility layer
│   ├── topology_extractor.py        # Edge & adjacency logic
│   └── config/
│       └── Config.yaml              # Configuration settings
```

## 🛠️ Installation

Install the required Python dependencies:

```bash
pip install numpy pyvista pyyaml scipy
```

## ⚙️ Configuration

Before running the pipeline, you must configure your input and output directories in the configuration file.

Open [utils/config/Config.yaml](utils/config/Config.yaml) and edit the paths according to your system:

```yaml
raw_input_dir: "path/to/raw_input"       # Directory containing .nas files
rafined_input_dir: "path/to/refined"     # Secondary input directory (optional)
raw_output_csv_dir: "path/to/output_raw" # Output for raw data CSVs
rafined_output_csv_dir: "path/to/output_refined" # Output for processed CSVs
summary_dir: "path/to/summary"           # Output for JSON summaries
logs_dir: "logs"
```

## ▶️ Usage

### 1. Data Processing Pipeline

Run `main.py` to execute the full analysis pipeline. This script reads `.nas` files from the input directory, analyzes properties, applies canonical ordering, and saves the results.

```bash
python main.py
```

### 2. Visualization Tools

You can visual inspection tools located in the `scripts/` folder.

**Note:** Ensure you update the `mesh_file` path variable inside the `if __name__ == "__main__":` block of these scripts before running.

*   **View Boundary Nodes:**
    ```bash
    python scripts/visualize_boundary_nodes.py
    ```

*   **View Node Indices & Connections:**
    ```bash
    python scripts/visualize_first_10_nodes.py
    ```

## 📊 Output Formats

The tool generates the following structured CSV files in your output directories:

*   **nodes/:** `node_id, x, y, z, normal_x, normal_y, normal_z, mean_curvature, gaussian_curvature, degree, is_boundary...`
*   **elements/:** `element_id, node_list, area, normal_vector...`
*   **edges/:** `edge_id, node1, node2, length...`
*   **mapping/:**
    *   `edge_to_id.csv`
    *   `element_neighbors.csv` (Adjacency list)
    *   `element_edges.csv` (Edges belonging to elements)

## 📝 License

This project is part of the MeshCrafter software suite.
```