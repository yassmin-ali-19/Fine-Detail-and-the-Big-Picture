import numpy as np

#_________________-using loops_____________________________--

def convolve2D_loop(image, kernel):

    #flip the kernel horizontally and vertically 
    kernel = np.flipud(np.fliplr(kernel))

    xKernShape = kernel.shape[0] 
    yKernShape = kernel.shape[1] 
    xImgShape = image.shape[0] 
    yImgShape = image.shape[1]

    xPadding = xKernShape //2
    yPadding = yKernShape //2


    xOtputShape = xImgShape
    yOutputShape = yImgShape
    output = np.zeros((xOtputShape, yOutputShape))

    # a padding of 1 increases the rows by 2 and the colunmns by 2
    imagePadded = np.zeros((xImgShape+ xPadding*2, yImgShape + yPadding*2))
    #select from the matix the image entries, start from index that's after the padding to the right before the padding
    imagePadded[xPadding:int(-1 * xPadding), yPadding:int(-1 * yPadding)] = image
    #print(imagePadded)

    # Iterate through image

    for x in range(xImgShape):           # every output row
        for y in range(yImgShape):       # every output column
            region = imagePadded[x:x + xKernShape, y:y + yKernShape]   # the window under the kernel
            output[x, y] = (kernel * region).sum()
    return output

#_____________________using vectorization____________________

def convole2D_vectorized(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    
    xKernShape = kernel.shape[0] 
    yKernShape = kernel.shape[1] 
    xImgShape = image.shape[0] 
    yImgShape = image.shape[1]

    xPadding = xKernShape //2
    yPadding = yKernShape //2


    xOtputShape = xImgShape
    yOutputShape = yImgShape
    output = np.zeros((xOtputShape, yOutputShape))

    imagePadded = np.zeros((xImgShape+ xPadding*2, yImgShape + yPadding*2))
    imagePadded[xPadding:int(-1 * xPadding), yPadding:int(-1 * yPadding)] = image
    

    for a in range(xKernShape):
        for b in range(yKernShape):
            output += kernel[a, b] * imagePadded[a:a + xImgShape, b:b + yImgShape]
    return output


#___________________seperable filters_________________-

def is_separable(kernel, tol=1e-10):

    #tolerance is used to deal with float number rounding errors, like le -4 is still considered >0

    U, S, Vt = np.linalg.svd(kernel)
    non_zero_singular_values = np.sum(S > tol)

    if non_zero_singular_values == 1:
        print("Kernel is separable (Rank 1).")
        # Extract 1D filters
        col = U[:, 0] * np.sqrt(S[0])
        row = Vt[0, :] * np.sqrt(S[0])
        print("Column filter:", col)
        print("Row filter:", row)
        return True, col, row
    else: 
        return False, None, None


def convolve1D(image, vec, axis):

    k = vec.shape[0]
    pad = k // 2
    vec_flipped = vec[::-1]

    if axis == 0:
        H, W = image.shape
        imagePadded = np.zeros((H + 2 * pad, W))
        imagePadded[pad:pad + H, :] = image
        output = np.zeros((H, W))
        for a in range(k):
            output += vec_flipped[a] * imagePadded[a:a + H, :]
    else:  # axis == 1
        H, W = image.shape
        imagePadded = np.zeros((H, W + 2 * pad))
        imagePadded[:, pad:pad + W] = image
        output = np.zeros((H, W))
        for b in range(k):
            output += vec_flipped[b] * imagePadded[:, b:b + W]
    return output


def convolve2D_separable(image, kernel):

    sep, col, row = is_separable(kernel)
    if not sep:
        return convole2D_vectorized(image, kernel)
    out = convolve1D(image, col, axis=0)   
    out = convolve1D(out, row, axis=1)     
    return out




#_______timing the functions using guassian with different sizes_________
#the results of comparison are written into csv, and many tries are made and best is taken

import csv
import time
import numpy as np
from scipy.signal import convolve2d

def gaussian_kernel(size, sigma=None):
    #computer sigma from size based on equation : k_w = 2(3sigma)+1
    sigma = sigma or size / 6.0
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)

def best_time(f, image, ker, repeats=5):
    #Minimum over several runs: least affected by other running processes noise
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        f(image, ker)
        times.append(time.perf_counter() - t)
    return min(times)

if __name__ == "__main__":


    image = np.random.default_rng(0).random((256, 256))

    funcs = {
        "loop": convolve2D_loop,
        "vectorized": convole2D_vectorized,
        "separable": convolve2D_separable,
    }

    rows = []
    for k in range(3, 32, 2):
        ker = gaussian_kernel(k)
        ref = convolve2d(image, ker, mode="same")

        row = {"kernel_size": f"{k}x{k}"}
        for name, f in funcs.items():
            row[f"{name}_s"] = best_time(f, image, ker, repeats=1 if name == "loop" else 5)
            row[f"{name}_max_err"] = np.abs(f(image, ker) - ref).max()
        row["scipy_s"] = best_time(lambda im, kr: convolve2d(im, kr, mode="same"), image, ker)
        rows.append(row)

    with open("timing_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved timing_results.csv with", len(rows), "rows")