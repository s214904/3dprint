from algorithms import compute_tenengrad, compute_brenner_gradient, compute_sobel_variance, compute_laplacian
def compute_sharpness(gray_img, method):
    if method == 'tenengrad':
        return compute_tenengrad(gray_img)
    elif method == 'brenner':
        return compute_brenner_gradient(gray_img)
    elif method == 'sobel_variance':
        return compute_sobel_variance(gray_img)
    elif method == 'laplacian':
        return compute_laplacian(gray_img)
    else:
        raise ValueError(f"Unknown sharpness metric: {method}")
