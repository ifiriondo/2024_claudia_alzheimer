
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

    return data_3d