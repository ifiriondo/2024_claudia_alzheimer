import os
import pandas as pd
import numpy as np
import xarray as xr

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