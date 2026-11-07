# NasMeshLib

Turns NASTRAN (`.nas`) surface meshes into machine-learning-ready graph data: nodes, edges and elements with geometric and topological features, in a canonical order so the same mesh always produces the same sequence.

It is the preprocessing stage of **MeshCrafter**, a generative model that learns to produce refined FE meshes from raw ones. For each part, a raw mesh and a refined mesh (same file name, two folders) are processed into paired CSV datasets.

## What it computes

| Level | Features |
| --- | --- |
| Node | coordinates, area-weighted normal, **mean curvature**, **Gaussian curvature**, degree, boundary flag |
| Edge | node pair, length |
| Element | node list (CTRIA3 / CQUAD4), area, normal |
| Graph | element adjacency, element ↔ edge maps, edge-to-id map |
| Mesh | bounding box, surface area, mean edge length, counts (JSON summary) |

**Canonical ordering.** Nodes are sorted spatially (Z → Y → X); elements are re-indexed by breadth-first traversal of the element adjacency graph. This removes the arbitrary IDs a mesher assigns, which matters for sequence models that generate elements one at a time.

**Curvature.** Discrete differential geometry on the triangulated surface (quads split into two triangles):

- Mean curvature: cotangent Laplace–Beltrami operator, `H = -½ ⟨Δx, n⟩`, signed with respect to the vertex normal.
- Gaussian curvature: angle deficit (`2π − Σθ`, `π` on boundary nodes) divided by the barycentric vertex area.

## Validation

`tests/test_curvature.py` checks the parser, topology and curvature against shapes with known answers:

| Shape | Mesh | Mean curvature H | Gaussian curvature K |
| --- | --- | --- | --- |
| Sphere, R = 10 | 3,000 nodes, triangles | 0.0999 (exact: 0.1) | 0.0100 (exact: 0.01) |
| Cylinder, R = 5 | 2,400 nodes, quads | 0.1000 (exact: 0.1) | 0.0000 (exact: 0) |
| Plane | triangles | 0 | 0 |

The topology test also checks Euler's relation for a closed triangle mesh (E = 3F/2).

> **Note:** versions before this one computed Gaussian curvature from an unordered neighbour set without dividing by area, and halved the mean curvature. The tests above were added with the fix; features generated with older versions should be regenerated.

## Installation

```bash
pip install numpy scipy pyyaml pyvista pytest
```

PyVista is only needed for boundary detection inside `find_boundary_nodes` and for the visualization scripts.

## Usage

1. Put raw meshes in one folder and refined meshes with the **same file names** in another.
2. Set the folders in [`utils/config/Config.yaml`](utils/config/Config.yaml) (or point `NASMESHLIB_CONFIG` to your own YAML):

   ```yaml
   raw_input_dir: "data/raw"
   rafined_input_dir: "data/refined"
   raw_output_csv_dir: "output/raw"
   rafined_output_csv_dir: "output/refined"
   summary_dir: "output/summary"
   logs_dir: "logs"
   ```

3. Run the pipeline and the tests:

   ```bash
   python main.py
   python -m pytest tests
   ```

Visualization:

```bash
python scripts/visualize_boundary_nodes.py path/to/mesh.nas
python scripts/visualize_first_10_nodes.py path/to/mesh.nas
```

## Output

```
output/
  raw/ and refined/
    nodes/<part>.csv        node_id, x, y, z, normals, mean/gaussian curvature, degree, is_boundary
    edges/<part>.csv        edge_id, node1, node2, length
    elements/<part>.csv     element_id, type, node list, normal, neighbours
    mapping/                edge_to_id, edge_elements, element_edges, element_neighbors
  summary/
    raw_summary.json, refined_summary.json
```

## Project structure

```
main.py                     raw → refined pair pipeline
utils/
  ParserNas.py              NASTRAN parser (GRID, CTRIA3, CQUAD4, continuation lines, malformed exponents)
  topology_extractor.py     edges, adjacency, element/edge maps
  geometry_features.py      normals, boundary, curvature
  canonical_order.py        spatial node sort + BFS element ordering
  mesh_summary.py           bounding box, area, edge statistics
  csv_utils.py              CSV / JSON export
  config/Config.yaml        paths
scripts/                    PyVista visualizations
tests/                      analytic validation
```

## License

Part of the MeshCrafter project.
