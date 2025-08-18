from algorithms import compute_tenengrad, compute_brenner_gradient, compute_sobel_variance, compute_laplacian
def compute_sharpness(gray_img, method):

    if 1 == 0:
        gray_img = _fft_magnitude_image(gray_img)

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

def _fft_magnitude_image(gray):
    import numpy as np
    """Return normalized log‑magnitude spectrum (float32 0..1)."""
    g = gray.astype(np.float32)
    g -= g.mean()
    # Optional window to reduce edge leakage
    wy = np.hanning(g.shape[0])
    wx = np.hanning(g.shape[1])
    g *= wy[:, None] * wx[None, :]
    F = np.fft.fft2(g)
    mag = np.abs(F)
    mag = np.fft.fftshift(mag)
    mag = np.log1p(mag)
    mag -= mag.min()
    mag /= (mag.max() + 1e-8)
    return mag  # float32 0..1