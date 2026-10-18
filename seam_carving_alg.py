from finite_difference import compute_edges
import numpy as np
from utils import load_image 
import matplotlib.pyplot as plt
from derivative_of_guassian import DOG

IMAGE_DIR1 = "D:/Documents/Fall 26/Computer Vision/Asg/Asg1/starter_pack/images/part3_seams/"
IMAGE_DIR2 = "D:/Documents/Fall 26/Computer Vision/Asg/Asg1/starter_pack/images/part3_mine/"

def to_gray(im):
    if im.ndim == 2:
        return im
    return im[..., 0] * 0.299 + im[..., 1] * 0.587 + im[..., 2] * 0.114

def find_cheapest_seam(energy_map):
    rows, cols = energy_map.shape
    cost = energy_map.astype(float).copy()
    backtrack = np.zeros((rows, cols), dtype=int)
    idx = np.arange(cols)

    for i in range(1, rows):
        prev = cost[i - 1]
        left = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        parents = np.stack([left, prev, right])      # shape (3, cols)
        choice = np.argmin(parents, axis=0)          # 0, 1, 2
        backtrack[i] = idx + choice - 1
        cost[i] += parents[choice, idx]

    seam = np.zeros(rows, dtype=int)
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    return seam

def remove_seam(arr, seam):
    rows, cols = arr.shape[:2]
    mask = np.ones((rows, cols), dtype=bool)
    mask[np.arange(rows), seam] = False 
    if arr.ndim == 2:
        return arr[mask].reshape(rows, cols - 1)
    return arr[mask].reshape(rows, cols - 1, arr.shape[2])

def carve_vertical(im, num_seams = 0, mode = "finite_difference"):
    H, W = im.shape[:2]

    if num_seams == 0:
        num_seams = int(round(H * 30 / 100))

    assert num_seams < W, "Dimension Error"
 
    carved = im.copy()
    col_map = np.tile(np.arange(W), (H, 1))      
    rows_idx = np.arange(H)
    seams_orig = []
 
    for _ in range(num_seams):
        if mode == "finite_difference":
            dIdx, dIdy, _, _, _ = compute_edges(to_gray(carved), mode="edge")
        else:
            _, _, dIdx, dIdy, _, _, _ = DOG(to_gray(carved))
        energy_map = np.sqrt(dIdx**2 + dIdy**2)
        seam = find_cheapest_seam(energy_map)
 
        seams_orig.append(col_map[rows_idx, seam].copy())  
        carved = remove_seam(carved, seam)              
        col_map = remove_seam(col_map, seam)            
 
    return carved, np.array(seams_orig)
 

def _T(im):
    return np.swapaxes(im, 0, 1)                 
 
def carve_horizontal(im, num_seams = 0, mode = "finite_difference"):
    carved_T, seams_orig = carve_vertical(_T(im), num_seams, mode=mode)
    return _T(carved_T), seams_orig
 
 
def draw_seams(im, seams_orig, color=(1.0, 0.0, 0.0), horizontal=False):
    out = im.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    if horizontal:
        out = _T(out).copy()          
 
    H = out.shape[0]
    rows_idx = np.arange(H)
    for seam in seams_orig:
        out[rows_idx, seam] = color
 
    return _T(out).copy() if horizontal else out


def show_comparison(im, resized, cropped, carved, path="comparison.png"):
    imgs = [im, resized, cropped, carved]
    titles = ["Original", "Resized", "Cropped", "Seam carved"]
    widths = [x.shape[1] for x in imgs]          # panel width ~ image width
    H = im.shape[0]

    fig_h = 3.0
    fig_w = sum(widths) / H * fig_h * 1.08       # a bit extra for the gaps
    fig, axs = plt.subplots(1, 4, figsize=(fig_w, fig_h + 0.4),
                            gridspec_kw={"width_ratios": widths, "wspace": 0.04})

    for ax, img, title in zip(axs, imgs, titles):
        ax.imshow(img)                            # add cmap="gray" only for 2D images
        ax.axis("off")
        ax.text(0.5, -0.04, title, transform=ax.transAxes,
                ha="center", va="top", fontsize=12)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def compare(im, name):
    import cv2

    n = int(round(0.3 * im.shape[1]))
    carved, seams = carve_vertical(im, n, mode="dog")
    new_W = carved.shape[1]

    resized = cv2.resize(im, (new_W, im.shape[0]), interpolation=cv2.INTER_AREA)
    x0 = (im.shape[1] - new_W) // 2
    cropped = im[:, x0:x0 + new_W]

    show_comparison(im, resized, cropped, carved, f"compare_{name}.png")


def test_failing_example():
    name = "pic4.jpg"
    im = load_image(f"{IMAGE_DIR2}/{name}")       
    carved, seams = carve_vertical(im, num_seams=0, mode="dog")
    with_seams = draw_seams(im, seams)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(im);         axs[0].set_title("Original")
    axs[1].imshow(with_seams); axs[1].set_title(f"{len(seams)} seams")
    axs[2].imshow(carved);     axs[2].set_title(f"Carved ({carved.shape[1]} cols)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"part3_mine_{name.split('.')[0]}.png", dpi=150)
    plt.close(fig)

names1 = ["broadway_tower.png", "home.jpg", "rocket.jpg", "starry_night.jpg"]
names2 = ["pic1.jpg", "pic2.jpg", "pic3.jpg"]

for name in names1:
    im = load_image(f"{IMAGE_DIR1}/{name}")       
    carved, seams = carve_vertical(im, num_seams=0, mode="dog")
    with_seams = draw_seams(im, seams)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(im);         axs[0].set_title("Original")
    axs[1].imshow(with_seams); axs[1].set_title(f"{len(seams)} seams")
    axs[2].imshow(carved);     axs[2].set_title(f"Carved ({carved.shape[1]} cols)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"part3_seams_{name.split('.')[0]}.png", dpi=150)
    plt.close(fig)
    compare(im, name.split('.')[0])

for name in names2:
    im = load_image(f"{IMAGE_DIR2}/{name}")       
    carved, seams = carve_vertical(im, num_seams=0, mode="dog")
    with_seams = draw_seams(im, seams)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(im);         axs[0].set_title("Original")
    axs[1].imshow(with_seams); axs[1].set_title(f"{len(seams)} seams")
    axs[2].imshow(carved);     axs[2].set_title(f"Carved ({carved.shape[1]} cols)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"part3_mine_{name.split('.')[0]}.png", dpi=150)
    plt.close(fig)
    compare(im, name.split('.')[0])

test_failing_example()