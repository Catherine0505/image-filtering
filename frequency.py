import numpy as np
import skimage as sk
import skimage.io as skio
import os
from skimage.transform import resize
import scipy
import cv2

gaussian_filter_1d = cv2.getGaussianKernel(ksize = 3, sigma = 2)
gaussian_filter = np.dot(gaussian_filter_1d, gaussian_filter_1d.T)

def sharpen(image, gaussian_filter, alpha):
    """
    Sharpens the image given a gaussian filter and weight alpha.
    :param image: the image that needs to be sharpened.
    :param gaussian_filter: the gaussian filter that needs to be applied on the
        image.
    :param alpha: comes from the formula: alpha * (image - blurred_image) + image.
        Set the weight of high-resolution portion of the image. The larger alpha
        is, the more sharpened the image will be.
    """
    filter_height, filter_width = gaussian_filter.shape
    identity = np.array([[0] * filter_width] * filter_height)
    identity[-1, -1] = 1
    # Set the sharpening filter.
    filter = identity * (1 + alpha) - alpha * gaussian_filter
    # If the image is gray-scale, apply sharpening filter directly to the image.
    if len(image.shape) == 2 or (len(image.shape) > 2 and image.shape[2] == 1):
        return scipy.signal.convolve2d(image, filter, mode = "valid")
    # If the image is colored, apply sharpening filter to R, G and B channels
    # respectively.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return np.dstack([scipy.signal.convolve2d(r, filter, mode = "valid"),
        scipy.signal.convolve2d(g, filter, mode = "valid"),
        scipy.signal.convolve2d(b, filter, mode = "valid")])

def main():
    # Sharpen "taj.jpeg"
    image = skio.imread("taj.jpeg")
    image_sharpened = sharpen(image, gaussian_filter, alpha = 1.5)
    skio.imshow(np.clip(image_sharpened / 255, 0, 1))
    skio.show()

    # Sharpen "big_sur.jpeg"
    image = skio.imread("big_sur.png")
    image_sharpened = sharpen(image, gaussian_filter, alpha = 1.5)
    skio.imshow(np.clip(image_sharpened / 255, 0, 1))
    skio.show()

    # First blur "rainbow.jpeg", and then sharpen the blurred image. 
    image = skio.imread("rainbow.jpeg")
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    filter = cv2.getGaussianKernel(ksize = 4, sigma = 2)
    filter = np.dot(filter, filter.T)
    image_blur = np.dstack([scipy.signal.convolve2d(r, filter, mode = "valid"),
        scipy.signal.convolve2d(g, filter, mode = "valid"),
        scipy.signal.convolve2d(b, filter, mode = "valid")])
    image_blur = np.clip(image_blur / 255, 0, 1)
    skio.imshow(image_blur)
    skio.show()
    image_sharpened = sharpen(image_blur, gaussian_filter, alpha = 1.5)
    skio.imshow(np.clip(image_sharpened, 0, 1))
    skio.show()

if __name__ == "__main__":
    main()
