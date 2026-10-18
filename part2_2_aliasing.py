"""
2.2 Aliasing

rings.png is shrunk to a quarter of its width and height two ways:
  1. naive   - just keep every 4th pixel in each direction
  2. blurred - Gaussian-blur first, then keep every 4th pixel

The first one aliases (false circles appear that were never in the
original image); the second one, with a big enough sigma, does not.
"""
import matplotlib.pyplot as plt

from convolution import gaussian_kernel, convolve2D_separable
from utils import load_image

RINGS_PATH = "/home/heba/grad_project/cv_practice/labs/lab1/starter_pack/images/part2_frequencies/rings.png"
FACTOR = 4      # shrink to a quarter of width and height
SIGMA = 3.5     # smallest sigma at which the false circles disappear (see sweep below)


def gaussian_blur(image, sigma):
    """Blur with a Gaussian sized to its sigma (the usual 6*sigma+1 rule)."""
    size = max(3, int(6 * sigma + 1))
    if size % 2 == 0:
        size += 1
    kernel = gaussian_kernel(size, sigma)
    return convolve2D_separable(image, kernel)


def shrink_naive(image, factor):
    """Just keep every `factor`-th pixel in each direction - no filtering first."""
    return image[::factor, ::factor]


def shrink_with_blur(image, factor, sigma):
    """Low-pass with a Gaussian first, then keep every `factor`-th pixel."""
    blurred = gaussian_blur(image, sigma)
    return blurred[::factor, ::factor]


def show_pair(images, titles, suptitle, out_path):
    fig, axes = plt.subplots(1, len(images), figsize=(4.5 * len(images), 4.5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    rings = load_image(RINGS_PATH, gray=True)
    h, w = rings.shape

    naive = shrink_naive(rings, FACTOR)
    smoothed = shrink_with_blur(rings, FACTOR, SIGMA)

    show_pair(
        [naive, smoothed],
        [f"Naive: every {FACTOR}th pixel", f"Blurred (\u03c3={SIGMA}) then every {FACTOR}th pixel"],
        f"rings.png, {w}\u00d7{h} \u2192 {w // FACTOR}\u00d7{h // FACTOR}",
        "part2_2_aliasing.png",
    )

    # A small sweep across sigma, just to show why SIGMA above was picked.
    sigmas = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 5.0]
    imgs = [naive if s == 0 else shrink_with_blur(rings, FACTOR, s) for s in sigmas]
    labels = ["no blur" if s == 0 else f"\u03c3={s}" for s in sigmas]
    show_pair(imgs, labels, "Finding the smallest \u03c3 that removes the false circles",
              "part2_2_sigma_sweep.png")


if __name__ == "__main__":
    main()
