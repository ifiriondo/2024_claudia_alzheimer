import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial.distance import pdist, squareform


def process_data(data_dir):
    # List of files in the directory
    files = sorted(os.listdir(data_dir))

    # Initialize an empty list to hold the data
    data_list = []

    # Loop to read all the files
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        data = pd.read_csv(file_path, sep='\t', header=0)
        data_list.append(data.values)

    # Convert the list to a 3D numpy array
    data_3d = np.array(data_list)

    # Add headers from the original data to data_3d
    headers = data.columns

    # Create the xarray DataArray
    data_array = xr.DataArray(data_3d, dims=['subject', 'timepoint', 'struc'], 
                            coords={'subject': [f'Subject{i:03d}' for i in range(1, len(data_3d) + 1)], 
                                    'timepoint': np.arange(np.shape(data_3d)[1]), 
                                    'struc': headers})

    return data_3d, data_array, headers


def calc_corr_matrix(data_array, headers):
    # Initialize an empty list to hold the correlation matrices
    correlation_matrices = []

    # Loop over all subjects to compute correlation matrices
    for sub in data_array.subject.values:
        specific_subject_data = data_array.sel(subject=sub)
        specific_subject_df = specific_subject_data.to_pandas()
        correlation_matrix = specific_subject_df.corr()
        correlation_matrices.append(correlation_matrix.values)

    # Convert the list of correlation matrices to a 3D numpy array
    correlation_3d = np.array(correlation_matrices)

    # Create coordinates for the new DataArray
    corr_coords = {
        'subject': [f'Subject{i:03d}' for i in range(1, len(correlation_3d) + 1)],
        'struc1': headers,
        'struc2': headers
    }

    # Create the xarray DataArray for the correlation matrices
    correlation_data_array = xr.DataArray(correlation_3d, dims=['subject', 'struc1', 'struc2'], coords=corr_coords)

    # Convert to xarray Dataset if needed
    correlation_dataset = xr.Dataset({'correlation_matrix': correlation_data_array})

    return correlation_3d, correlation_dataset, correlation_data_array

import numpy as np
import pandas as pd

def split_into_windows_and_compute_correlation(data, delta):
    num_subjects, num_timepoints, num_structures = data.shape
    num_windows = num_timepoints // delta
    
    # Inicializar un array para almacenar las matrices de correlación
    corr_matrices = np.zeros((num_structures, num_structures, num_subjects, num_windows))
    
    # Calcular las matrices de correlación para cada ventana
    for subj in range(num_subjects):
        for win in range(num_windows):
            start_idx = win * delta
            end_idx = start_idx + delta
            window_data = data[subj, start_idx:end_idx, :]
            
            # Calcular la matriz de correlación para la ventana actual
            corr_matrix = np.corrcoef(window_data, rowvar=False)
            corr_matrices[:, :, subj, win] = corr_matrix
    
    return corr_matrices




import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_norm_distances(corr_matrices_structure):
    num_structures, num_subjects, num_windows = corr_matrices_structure.shape
    
    # Inicializar un array para almacenar las matrices de distancias
    norm_distances = np.zeros((num_windows, num_windows, num_subjects))
    
    for subj in range(num_subjects):
        # Extraer las ventanas para el sujeto actual
        windows = corr_matrices_structure[:, subj, :].T  # Transponer para obtener (33, 61)
        
        # Calcular las distancias de pares
        dist_matrix = squareform(pdist(windows, 'euclidean'))
        norm_distances[:, :, subj] = dist_matrix
    
    return norm_distances

def flatten_distance_matrices(norm_distances):
    num_windows, _, num_subjects = norm_distances.shape
    
    # Inicializar un array para almacenar los vectores aplanados
    flattened_distances = np.zeros((num_subjects, num_windows * num_windows))
    
    for subj in range(num_subjects):
        # Aplanar la matriz de distancias para el sujeto actual
        flattened_distances[subj, :] = norm_distances[:, :, subj].flatten()
    
    return flattened_distances













