import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from utils import load_image
from finite_difference import compute_edges
from seam_carving import to_gray, find_cheapest_seam, remove_seam, carve_vertical, draw_seams

PHOTO_PATH = "/home/heba/grad_project/cv_practice/labs/lab1/starter_pack/images/part3_seams/broadway_tower.png"
MASK_PATH = "/home/heba/grad_project/cv_practice/labs/lab1/starter_pack/code/human_mask.png"
BIG_NEGATIVE = -1e6
MASK_DILATION = 2   # grows the hand-painted mask by a couple of pixels so it
                     # also covers the object's soft/antialiased edge - without
                     # this a thin fringe of blended edge pixels gets left behind


def energy_map(gray, mode="edge"):
    dIdx, dIdy, _, _, _ = compute_edges(gray.astype(np.float64), mode=mode)
    return np.sqrt(dIdx ** 2 + dIdy ** 2)


def remove_object(photo, mask, mode="edge"):
    """Carve vertical seams through the masked object until it's gone.
    Returns the object-free photo and k, the number of seams removed."""
    carved = photo.copy()
    mask_c = mask.copy().astype(bool)
    k = 0
    while mask_c.any():
        energy = energy_map(to_gray(carved), mode=mode)
        energy[mask_c] = BIG_NEGATIVE
        seam = find_cheapest_seam(energy)
        carved = remove_seam(carved, seam)
        mask_c = remove_seam(mask_c, seam)
        k += 1
    return carved, k


def insert_seams(photo, seams_orig):
    """Widen `photo` by len(seams_orig) columns: every pixel on one of the
    given seams is written twice (averaged with its left, then its right
    neighbor); every other pixel is written once, unchanged."""
    H, W = photo.shape[:2]
    k = seams_orig.shape[0]
    new_w = W + k
    out = np.zeros((H, new_w, photo.shape[2]), dtype=photo.dtype)
    inserted = np.zeros((H, new_w), dtype=bool)

    for i in range(H):
        seam_cols = sorted(seams_orig[:, i].tolist())
        seam_set = set(seam_cols)
        out_col = 0
        for j in range(W):
            if j in seam_set:
                left = photo[i, j - 1] if j - 1 >= 0 else photo[i, j]
                right = photo[i, j + 1] if j + 1 < W else photo[i, j]
                out[i, out_col] = (photo[i, j] + left) / 2.0
                inserted[i, out_col] = True
                out_col += 1
                out[i, out_col] = (photo[i, j] + right) / 2.0
                inserted[i, out_col] = True
                out_col += 1
            else:
                out[i, out_col] = photo[i, j]
                out_col += 1
    return out, inserted


def overlay_mask(photo, mask, color=(1.0, 0.0, 0.0), alpha=0.6):
    out = photo.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return out


def main():
    photo = load_image(PHOTO_PATH)                                   # RGB float
    mask_painted = load_image(MASK_PATH, gray=True) > 0.5             # bool, as painted
    mask = binary_dilation(mask_painted, iterations=MASK_DILATION)    # + a couple px margin
                                                                        # so the object's soft edge
                                                                        # is covered too

    # 1. object removal
    reduced, k = remove_object(photo, mask)
    print(f"Removed object with k = {k} vertical seams "
          f"({photo.shape[1]} -> {reduced.shape[1]} columns)")

    # 2. grow back by k seams, found on a throwaway copy
    copy_for_insertion = reduced.copy()
    _, seams_orig = carve_vertical(copy_for_insertion, k)
    widened, inserted = insert_seams(reduced, seams_orig)
    assert widened.shape[1] == photo.shape[1], "widened photo should match the original width"

    widened_with_seams = widened.copy()
    widened_with_seams[inserted] = (1.0, 0.0, 0.0)

    # ---- figure: mask, object removed, regrown with seams drawn ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(overlay_mask(photo, mask_painted))
    axs[0].set_title("Photo with its mask")
    axs[1].imshow(reduced)
    axs[1].set_title(f"Object removed ({reduced.shape[1]} cols, k={k})")
    axs[2].imshow(widened_with_seams)
    axs[2].set_title(f"Regrown to {widened.shape[1]} cols, inserted seams shown")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("bonus_object_removal.png", dpi=150)
    print("saved bonus_object_removal.png")


if __name__ == "__main__":
    main()
