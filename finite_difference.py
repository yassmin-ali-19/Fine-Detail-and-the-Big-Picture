import numpy as np
import matplotlib.pyplot as plt
from convolution import convolve2D_separable  #the deffernce vectors are already 1D, so svd would return true
from utils import load_image


def compute_edges(im, k=2.0, mode="zero"):
    #the difference vectors:
    Dx = np.array([[1.0, 0.0, -1.0]])   # H vector
    Dy = Dx.T                           # V

    #filter with the separable convolution
    dIdx = convolve2D_separable(im, Dx, mode=mode)
    dIdy = convolve2D_separable(im, Dy, mode=mode)

    #gradient magnitude:
    magnitude = np.sqrt(dIdx**2 + dIdy**2)

    #threshold -> edge map, it was chosed in a way to account for mean and variance of magnitude values, 2 here is an adjusted hyperparameter
    threshold = magnitude.mean() + k * magnitude.std()
    edges = (magnitude > threshold).astype(float)

    return dIdx, dIdy, magnitude, edges, threshold


#maps the the range of dI/dX, and dI/dY form [-1,1] to [0,1]
def zero_centered(im, clip=None):
    m = clip if clip is not None else np.max(np.abs(im))  #get the max value in img, whether positive or negative
    m = m if m > 0 else 1.0
    return np.clip(im / (2 * m) + 0.5, 0, 1)


if __name__ == "__main__":
    IMAGE_DIR = "/home/user/uploads"
    im = load_image(f"{IMAGE_DIR}/cameraman.png", gray=True) if False else None
    # (demo/plotting block kept out of the way of importing compute_edges elsewhere)
