import numpy as np
import matplotlib.pyplot as plt
from convolution import convolve2D_separable, gaussian_kernel
from utils import load_image

IMAGE_DIR = "D:/Documents/Fall 26/Computer Vision/Asg/Asg1/starter_pack/images/part1_filters/"
im = load_image(f"{IMAGE_DIR}/cameraman.png", gray=True)

#build the Gaussian kernel
SIGMA = 2.0
ksize = int(6 * SIGMA + 1)          # odd size covering ~+-3 sigma, same rule as convolution.py
G = gaussian_kernel(ksize, SIGMA)   # sums to 1, shape (ksize, ksize)

Dx = np.array([[1.0, 0.0, -1.0]])
Dy = Dx.T

DoG_x = convolve2D_separable(G, Dx)   # blur + x-derivative
DoG_y = convolve2D_separable(G, Dy)   # blur + y-derivative

# apply each DoG filter to the image
dIdx = convolve2D_separable(im, DoG_x)
dIdy = convolve2D_separable(im, DoG_y)

# gradient magnitude + threshold -> edges
magnitude = np.sqrt(dIdx**2 + dIdy**2)
threshold = magnitude.mean() + 1.5 * magnitude.std()
edges = (magnitude > threshold).astype(float)

print(f"sigma = {SIGMA}, kernel size = {ksize}x{ksize}")
print(f"magnitude: min={magnitude.min():.4f} max={magnitude.max():.4f} "
      f"mean={magnitude.mean():.4f} std={magnitude.std():.4f}")
print(f"threshold = {threshold:.4f}  -> {edges.mean()*100:.2f}% of pixels marked as edges")


def zero_centered(a):
    m = np.max(np.abs(a))
    m = m if m > 0 else 1.0
    return np.clip(a / (2 * m) + 0.5, 0, 1)


#figures:
#the derivative of guassian filters:
fig1, axs1 = plt.subplots(1, 2, figsize=(8, 4))
axs1[0].imshow(zero_centered(DoG_x), cmap="gray")
axs1[0].set_title(f"DoG in x, sigma={SIGMA}")
axs1[1].imshow(zero_centered(DoG_y), cmap="gray")
axs1[1].set_title(f"DoG in y, sigma={SIGMA}")
for ax in axs1:
    ax.axis("off")
plt.tight_layout()
plt.savefig("part1_3_filters.png", dpi=150)

#the four maps
fig2, axs2 = plt.subplots(1, 4, figsize=(16, 4.5))
axs2[0].imshow(zero_centered(dIdx), cmap="gray")
axs2[0].set_title(r"DoG-x response ($\partial I/\partial x$, blurred)")
axs2[1].imshow(zero_centered(dIdy), cmap="gray")
axs2[1].set_title(r"DoG-y response ($\partial I/\partial y$, blurred)")
axs2[2].imshow(magnitude, cmap="gray")
axs2[2].set_title("Gradient magnitude")
axs2[3].imshow(edges, cmap="gray")
axs2[3].set_title(f"Edges (t={threshold:.3f})")
for ax in axs2:
    ax.axis("off")
plt.tight_layout()
plt.savefig("part1_3_maps.png", dpi=150)
print("saved part1_3_filters.png and part1_3_maps.png")
