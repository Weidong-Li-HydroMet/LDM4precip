from torch.utils.data import DataLoader, Dataset
import numpy as np

def random_sample_fixed(matrix, lon_idx, lat_idx, r):
    if matrix.ndim == 2:
        sampled_matrix = matrix[int(lat_idx-r):int(lat_idx+r), 
                                int(lon_idx-r):int(lon_idx+r)]
    else:
        sampled_matrix = matrix[:, int(lat_idx-r):int(lat_idx+r), 
                                int(lon_idx-r):int(lon_idx+r)]
    return sampled_matrix

