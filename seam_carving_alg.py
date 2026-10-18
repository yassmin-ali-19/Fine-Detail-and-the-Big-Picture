from finite_difference import compute_edges
import numpy as np
from utils import load_image 
import matplotlib.pyplot as plt

IMAGE_DIR = "D:/Documents/Fall 26/Computer Vision/Asg/Asg1/starter_pack/images/part3_seams/"
im = load_image(f"{IMAGE_DIR}/broadway_tower.png", gray=True)

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

def carve_vertical(im, num_seams):
    H, W = im.shape[:2]
    assert num_seams < W
 
    carved = im.copy()
    col_map = np.tile(np.arange(W), (H, 1))      
    rows_idx = np.arange(H)
    seams_orig = []
 
    for _ in range(num_seams):
        dIdx, dIdy, _, _, _ = compute_edges(to_gray(carved), mode="edge")
        energy_map = np.sqrt(dIdx**2 + dIdy**2)
        seam = find_cheapest_seam(energy_map)
 
        seams_orig.append(col_map[rows_idx, seam].copy())  
        carved = remove_seam(carved, seam)              
        col_map = remove_seam(col_map, seam)            
 
    return carved, np.array(seams_orig)
 

def _T(im):
    return np.swapaxes(im, 0, 1)                 
 
def carve_horizontal(im, num_seams):
    carved_T, seams_orig = carve_vertical(_T(im), num_seams)
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

 
n = 100
carved, seams = carve_vertical(im, n)
with_seams = draw_seams(im, seams)
 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(im);         axs[0].set_title("Original")
axs[1].imshow(with_seams); axs[1].set_title(f"{n} seams")
axs[2].imshow(carved);     axs[2].set_title(f"Carved ({carved.shape[1]} cols)")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.savefig("part3_seams.png", dpi=150)