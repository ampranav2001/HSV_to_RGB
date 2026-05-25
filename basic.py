import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
import time

def bgr_to_hsv_image_numpy(image):
    # normalize the full image at once

    start_time = time.time()

    hsv_image = image.astype(np.float32) / 255.0

    # slicing bgr matrices
    b = hsv_image[:, :, 0]
    g = hsv_image[:, :, 1]
    r = hsv_image[:, :, 2]

    # get the highest and minimum

    # color_max = np.maximum(np.maximum(r, g), b)  # uses 2 np.maximum for keeping it as 2d
    # color_min = np.minimum(np.minimum(r, g), b)

    # axis =2 means , colour channel , it stops at each pixel coordinate and checks the color channel and picks the
    # maximum and minimum
    color_max = np.max(hsv_image, axis=2)
    color_min = np.min(hsv_image, axis=2)

    delta = color_max - color_min

    # value
    v = color_max

    # saturation
    # create a empty matrix to store
    s = np.zeros_like(color_max)

    # use of masks : to find where color_max is > 0 , means it is inside
    mask_s = color_max > 0
    # apply for the pixel inside the mask
    s[mask_s] = delta[mask_s] / color_max[mask_s]

    # Hue

    h = np.zeros_like(color_max)

    mask_r = (delta > 0) & (color_max == r)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)

    mask_g = (delta > 0) & (color_max == g)
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)

    mask_b = (delta > 0) & (color_max == b)
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

    # calculation for cv

    h_final = np.round(h / 2)  # 8 bit = 255 , h= 360 ,  360>255  246.9 -247
    s_final = np.round(s * 255)  # 0.0 -1.0    246.9 -246
    v_final = np.round(v * 255)

    hsv_numpy_image = np.dstack((h_final, s_final, v_final))

    elapsed_time = time.time()

    final_time = elapsed_time - start_time

    print("Method 2 (Basic NumPy) Time: {:.3f}s\n\n".format(final_time))

    return hsv_numpy_image
