import numpy as np
import scipy
from scipy import stats
import skimage as sk
import skimage.io as skio
import os
from skimage.transform import resize
import scipy
import cv2
from stack import gaussian_stack
from stack import laplacian_stack

def blending(image1, image2, gaussian_factor, laplacian_factor, mask):
    """
    Blends two images together with regards to a particular mask.
    :param image1: the first image that participates in blending.
    :param image2: the second image that participates in blending.
    :param gaussian_factor: the gaussian factor that is applied to the mask to
        smooth boundaries.
    :laplacian_factor: the gaussian factor used to create laplacian stack for
        the two images.
    :mask: the mask that indicates how to blend the two images.
    """
    gaussian_mask = gaussian_stack(mask, 6, gaussian_factor)
    laplacian_image1 = laplacian_stack(image1, 6, laplacian_factor)
    laplacian_image2 = laplacian_stack(image2, 6, laplacian_factor)
    result = []
    for i in range(len(gaussian_mask)):
        result_it = gaussian_mask[i] * laplacian_image1[i] + \
            (1 - gaussian_mask[i]) * laplacian_image2[i]
        result.append(result_it)
    result = np.array(result)
    result = np.sum(result, axis = 0)
    return result

def blending_apple_color():
    image1 = skio.imread("apple.jpeg")
    image1 = sk.img_as_float(image1)

    image2 = skio.imread("orange.jpeg")
    image2 = sk.img_as_float(image2)

    mask = np.zeros(image1.shape)
    # The apple image takes up left half of the image, while the orange takes up
    # the right half.
    mask[:, :image1.shape[0] // 2] = 1
    result = blending(image1, image2, 20, 8, mask)
    # Normalize the values in the array to [0, 1] to be visualized.
    result = np.dstack(
        [(result[:, :, 0] -  np.min(result[:, :, 0]))/ \
        (np.max(result[:, :, 0]) - np.min(result[:, :, 0])),
        (result[:, :, 1] -  np.min(result[:, :, 1]))/ \
        (np.max(result[:, :, 1]) - np.min(result[:, :, 1])),
        (result[:, :, 2] -  np.min(result[:, :, 2]))/ \
        (np.max(result[:, :, 2]) - np.min(result[:, :, 2]))
        ])
    skio.imshow(result)
    skio.show()

def blending_apple_gray():
    # Load "apple.jpeg" and "orange.jpeg" as gray-scale images.
    image1 = skio.imread("apple.jpeg", as_gray = True)
    image1 = sk.img_as_float(image1)

    image2 = skio.imread("orange.jpeg", as_gray = True)
    image2 = sk.img_as_float(image2)

    # The apple image takes up left half of the image, while the orange takes up
    # the right half.
    mask = np.zeros(image1.shape)
    mask[:, :image1.shape[0] // 2] = 1
    result = blending(image1, image2, 20, 8, mask)
    skio.imshow(result)
    skio.show()

def blending_pizza():
    image1 = skio.imread("pizza1.jpeg")
    image1 = sk.img_as_float(image1)[:, :, :3]

    image2 = skio.imread("pizza2.png")
    image2 = sk.img_as_float(image2)[:, :, :3]
    image2 = resize(image2, image1.shape, anti_aliasing = True)

    # Blur both images a little bit to enhance blending effect.
    blur_mask = cv2.getGaussianKernel(ksize = 8, sigma = 4)
    blur_mask = np.dot(blur_mask, blur_mask.T)
    image1 = np.dstack(
        [scipy.signal.convolve2d(image1[:, :, 0], blur_mask, mode = "same"),
        scipy.signal.convolve2d(image1[:, :, 1], blur_mask, mode = "same"),
        scipy.signal.convolve2d(image1[:, :, 2], blur_mask, mode = "same")])
    image2 = np.dstack(
        [scipy.signal.convolve2d(image2[:, :, 0], blur_mask, mode = "same"),
        scipy.signal.convolve2d(image2[:, :, 1], blur_mask, mode = "same"),
        scipy.signal.convolve2d(image2[:, :, 2], blur_mask, mode = "same")])
    image1 = np.dstack(
        [(image1[:, :, 0] -  np.min(image1[:, :, 0]))/ \
        (np.max(image1[:, :, 0]) - np.min(image1[:, :, 0])),
        (image1[:, :, 1] -  np.min(image1[:, :, 1]))/ \
        (np.max(image1[:, :, 1]) - np.min(image1[:, :, 1])),
        (image1[:, :, 2] -  np.min(image1[:, :, 2]))/ \
        (np.max(image1[:, :, 2]) - np.min(image1[:, :, 2]))
        ])
    image2 = np.dstack(
        [(image2[:, :, 0] -  np.min(image2[:, :, 0]))/ \
        (np.max(image2[:, :, 0]) - np.min(image2[:, :, 0])),
        (image2[:, :, 1] -  np.min(image2[:, :, 1]))/ \
        (np.max(image2[:, :, 1]) - np.min(image2[:, :, 1])),
        (image2[:, :, 2] -  np.min(image2[:, :, 2]))/ \
        (np.max(image2[:, :, 2]) - np.min(image2[:, :, 2]))
        ])

    # Pizza 1 takes the left half of the image, while pizza 2 takes the right
    # half.
    mask = np.zeros(image1.shape)
    mask[:, :image1.shape[0] // 2] = 1
    result = blending(image1, image2, 30, 16, mask)
    result = np.dstack(
        [(result[:, :, 0] -  np.min(result[:, :, 0]))/ \
        (np.max(result[:, :, 0]) - np.min(result[:, :, 0])),
        (result[:, :, 1] -  np.min(result[:, :, 1]))/ \
        (np.max(result[:, :, 1]) - np.min(result[:, :, 1])),
        (result[:, :, 2] -  np.min(result[:, :, 2]))/ \
        (np.max(result[:, :, 2]) - np.min(result[:, :, 2]))
        ])
    skio.imshow(result)
    skio.show()

def blending_squirrel():
    # Load and resize both images to reduce disk space.
    image1 = skio.imread("right_squirrel.jpeg")
    image1 = sk.img_as_float(image1)[:, :, :3]
    image1 = resize(image1, (image1.shape[0] // 2, image1.shape[1] // 2),
        anti_aliasing = True)
    image2 = skio.imread("left_squirrel.jpeg")
    image2 = sk.img_as_float(image2)[:, :, :3]
    image2 = resize(image2, (image2.shape[0] // 2, image2.shape[1] // 2),
        anti_aliasing = True)

    # Pad the first image to be of the same size as the second image.
    image1_pad = np.zeros(image2.shape)
    image1_pad[65:65 + image1.shape[0], -25 - image1.shape[1]: -25, :] = image1
    skio.imshow(image1_pad)
    skio.show()
    xs = np.arange(0, image2.shape[1])
    ys = np.arange(0, image2.shape[0])
    center = [65 + image1.shape[0] // 2, -25 - image1.shape[1] // 2 + image2.shape[1]]
    radius = image1.shape[1] * 0.5 + 2
    x_grid, y_grid = np.meshgrid(xs, ys)

    # Create a mask that centers around the right squirrel, excluding as much
    # background as possible to make synthesized image authentic.
    mask_1 = (y_grid - center[0]) ** 2 + (x_grid - center[1]) ** 2 <= radius ** 2
    mask_2 = (x_grid >= -25 - image1.shape[1] + image2.shape[1]) & \
        (x_grid <= -25 + image2.shape[1])
    mask_3 = (y_grid >= 65) & (y_grid <= 65 + image1.shape[0])
    mask = mask_1 & mask_2 & mask_3
    # Show the mask.
    skio.imshow(mask, cmap = "gray")
    skio.show()
    result = blending(image1_pad, image2, 100, 10, np.dstack([mask, mask, mask]))
    result = np.dstack(
        [(result[:, :, 0] -  np.min(result[:, :, 0]))/ \
        (np.max(result[:, :, 0]) - np.min(result[:, :, 0])),
        (result[:, :, 1] -  np.min(result[:, :, 1]))/ \
        (np.max(result[:, :, 1]) - np.min(result[:, :, 1])),
        (result[:, :, 2] -  np.min(result[:, :, 2]))/ \
        (np.max(result[:, :, 2]) - np.min(result[:, :, 2]))
        ])
    skio.imshow(result)
    skio.show()

if __name__ == "__main__":
    blending_apple_gray()
    blending_apple_color()
    blending_pizza()
    blending_squirrel()
