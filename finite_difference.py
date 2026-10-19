import numpy as np
import matplotlib.pyplot as plt
from convolution import convolve2D_separable  #the deffernce vectors are already 1D, so svd would return true
from utils import load_image 

IMAGE_DIR = "D:/Documents/Fall 26/Computer Vision/Asg/Asg1/starter_pack/images/part1_filters/"
im = load_image(f"{IMAGE_DIR}/cameraman.png", gray=True)

def compute_edges(im, k=2.0, mode = "zero"):
    #the difference vectors:
    Dx = np.array([[1.0, 0.0, -1.0]])   # H vector
    Dy = Dx.T                           # V

    #filter with the separable convolution
    dIdx = convolve2D_separable(im, Dx, mode = mode) 
    dIdy = convolve2D_separable(im, Dy, mode = mode)

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

def main():
    dIdx, dIdy, magnitude, edges, threshold = compute_edges(im, k=2.0)

    print(f"magnitude: min={magnitude.min():.3f} max={magnitude.max():.3f} "
        f"mean={magnitude.mean():.3f} std={magnitude.std():.3f}")
    print(f"chosen threshold = {threshold:.4f}  "
        f"-> {edges.mean()*100:.2f}% of pixels marked as edges")

    #show the four maps (+ a zoomed corner of the edges)
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].imshow(zero_centered(dIdx), cmap="gray")
    axs[0].set_title(r"$\partial I/\partial x$")

    axs[1].imshow(zero_centered(dIdy), cmap="gray")
    axs[1].set_title(r"$\partial I/\partial y$")

    axs[2].imshow(magnitude, cmap="gray")
    axs[2].set_title("Gradient magnitude")

    axs[3].imshow(edges, cmap="gray")
    axs[3].set_title(f"Edges (t={threshold:.3f})")

    crop = 20  # pixels to zoom on small one line oixel edges
    axs[4].imshow(edges[:crop, :crop], cmap="gray", interpolation="nearest")  #use interpolation here to not blend into zeros when stretched
    axs[4].set_xticks(np.arange(-.5, crop, 1), minor=True)
    axs[4].set_yticks(np.arange(-.5, crop, 1), minor=True)
    axs[4].grid(which="minor", color="red", linewidth=0.3)
    axs[4].set_xticks([]); axs[4].set_yticks([])
    axs[4].set_title("Edges, corner enlarged")

    for ax in axs[:4]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("part1_2_maps.png", dpi=150)
    print("saved part1_2_maps.png")