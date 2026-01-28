"""
PyVista version uyumluluğu için yardımcı fonksiyonlar.
"""

import pyvista as pv
import numpy as np


def get_cells_dict(mesh):
    """
    PyVista sürümüne göre cells_dict döndür.
    Format: {cell_type: [array of node indices]}
    """
    try:
        return mesh.cells_dict
    except AttributeError:
        # Yeni sürüm: mesh.faces veya mesh.regular_faces kullan
        cells_dict = {}
        
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            faces = mesh.faces
        elif hasattr(mesh, 'regular_faces'):
            faces = mesh.regular_faces
        else:
            return cells_dict
        
        i = 0
        while i < len(faces):
            n_nodes = int(faces[i])
            cell_indices = faces[i+1:i+n_nodes+1].astype(int)
            
            if n_nodes == 3:
                cell_type = "TRIANGLE"
            elif n_nodes == 4:
                cell_type = "QUAD"
            else:
                cell_type = "POLYGON"
            
            if cell_type not in cells_dict:
                cells_dict[cell_type] = []
            
            cells_dict[cell_type].append(cell_indices)
            i += n_nodes + 1
        
        return cells_dict


def get_n_cells(mesh):
    """PyVista sürümüne göre hücre sayısını döndür."""
    try:
        return len(mesh.cells_dict)
    except AttributeError:
        return mesh.n_cells