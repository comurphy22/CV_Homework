import os
from PIL import Image
import numpy as np
from scipy.io import loadmat
from scipy import signal


def load_and_preprocess_image(path, size):
    """Load an image, convert to grayscale, resize to (size,size) and return a float array [0,1]."""
    im = Image.open(path).convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=float) / 255.0
    return arr


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_filterbank(mat):
    """Find filter bank in loaded .mat file by trying common variable names."""
    for k in ("filterBank", "filters", "F", "LMfilters", "bank", "filter_bank", "fb"):
        if k in mat:
            return mat[k]
    # fallback: look for an ndarray with 3 dims and a dimension of 48
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim in (3, 4):
            if 48 in v.shape:
                return v
    return None


def interpret_filters_array(arr):
    """Convert filter bank array to list of 2D filters."""
    if arr is None:
        return []
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim == 3:
        h, w, n = arr.shape
        if n == 48:
            return [arr[:, :, i] for i in range(n)]
        if h == 48:
            return [arr[i, :, :] for i in range(h)]
    if arr.ndim == 4:
        arr_s = np.squeeze(arr)
        return interpret_filters_array(arr_s)
    # fallback: try to reshape to (h,w,48)
    flat = arr.ravel()
    if flat.size % 48 == 0:
        per = flat.size // 48
        side = int(np.sqrt(per))
        if side * side == per:
            resh = flat.reshape((side, side, 48))
            return [resh[:, :, i] for i in range(48)]
    raise ValueError("Could not interpret filter bank shape: {}".format(arr.shape))


def load_filters(filters_mat_path):
    """Load and return list of 2D filter arrays from .mat file."""
    mat = loadmat(filters_mat_path)
    fb = find_filterbank(mat)
    if fb is None:
        raise ValueError("Could not find filter bank in .mat file")
    return interpret_filters_array(fb)


def compute_filter_responses(image, filters):
    """Compute responses of image to all filters.
    
    Args:
        image: 2D numpy array (grayscale image)
        filters: list of 2D numpy arrays (filter kernels)
    
    Returns:
        responses: list of 2D numpy arrays (filter responses)
    """
    responses = []
    for filt in filters:
        filt = np.squeeze(np.array(filt, dtype=float))
        response = signal.convolve2d(image, filt, mode='same', boundary='symm')
        responses.append(response)
    return responses


def texture_repr(responses):
    """Compute texture representation from filter responses.
    
    Args:
        responses: list of 2D arrays (filter responses)
    
    Returns:
        repr_vector: 1D numpy array (texture representation)
    """
    # Compute mean and standard deviation for each filter response
    features = []
    for response in responses:
        features.append(np.mean(response))
        features.append(np.std(response))
    return np.array(features)


def texture_repr_concat(images, filters):
    """Compute concatenated texture representation for multiple images.
    
    Args:
        images: list of 2D numpy arrays (grayscale images)
        filters: list of 2D numpy arrays (filter kernels)
    
    Returns:
        concat_repr: 1D numpy array (concatenated representation)
    """
    all_features = []
    for image in images:
        responses = compute_filter_responses(image, filters)
        repr_vec = texture_repr(responses)
        all_features.extend(repr_vec)
    return np.array(all_features)


def texture_repr_mean(images, filters):
    """Compute mean texture representation for multiple images.
    
    Args:
        images: list of 2D numpy arrays (grayscale images)
        filters: list of 2D numpy arrays (filter kernels)
    
    Returns:
        mean_repr: 1D numpy array (mean representation)
    """
    all_reprs = []
    for image in images:
        responses = compute_filter_responses(image, filters)
        repr_vec = texture_repr(responses)
        all_reprs.append(repr_vec)
    return np.mean(all_reprs, axis=0)
