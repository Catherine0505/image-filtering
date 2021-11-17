import numpy as np
import skimage as sk
import skimage.io as skio
import os
from skimage.transform import resize
import scipy
import cv2

# Set the x derivative filter and y derivative filter.
x_filter = np.array([[1, -1]])
y_filter = np.array([[1], [-1]])

def derivative(image):
    """
    Finds the x-derivative and y-derivative of a given image.
    :param image: the image whose derivative needs to be calculated.
    """
    x_derivative = scipy.signal.convolve2d(image, x_filter, mode = "same")
    y_derivative = scipy.signal.convolve2d(image, y_filter, mode = "same")
    return x_derivative, y_derivative

# Calculates and visualizes the x-derivative and y-derivative of the original
# image.
image = skio.imread("cameraman.png", as_gray=True)
image = sk.img_as_float(image)
x_derivative, y_derivative = derivative(image)
x_derivative_normalize = (np.abs(x_derivative) > 0.09)
y_derivative_normalize = (np.abs(y_derivative) > 0.09)
skio.imshow(x_derivative_normalize)
skio.show()
skio.imshow(y_derivative_normalize)
skio.show()

# Calculates and visualizes the x-derivative and y-derivative of the Gaussian
# blurred image.
# First blurs the image using Gaussian filter, and then apply x_filter and y_filter
# to calculate the derivative of the blurred image.
gaussian_kernel_1d = cv2.getGaussianKernel(ksize = 3, sigma = -1)
gaussian_kernel = np.dot(gaussian_kernel_1d, gaussian_kernel_1d.T)
image_blurr = scipy.signal.convolve2d(image, gaussian_kernel, mode = "same")
skio.imshow(image_blurr)
skio.show()
x_derivative, y_derivative = derivative(image_blurr)
x_derivative_normalize = (np.abs(x_derivative) > 0.09)
y_derivative_normalize = (np.abs(y_derivative) > 0.09)
skio.imshow(x_derivative_normalize)
skio.show()
skio.imshow(y_derivative_normalize)
skio.show()

# Calculates and visualizes the x-derivative and y-derivative of the Gaussian
# blurred image.
# Calculates the x_derivative and y_derivative of the Gaussian kernel, and then
# convolves the result with the original image.
# Calculation of the derivative of Gaussian kernel should be precise, therefore,
# the *mode* parameter cannot be "same". 
gaussian_x_derivative = scipy.signal.convolve2d(gaussian_kernel, x_filter)
gaussian_y_derivative = scipy.signal.convolve2d(gaussian_kernel, y_filter)
skio.imshow(gaussian_x_derivative)
skio.show()
skio.imshow(gaussian_y_derivative)
skio.show()
x_derivative = scipy.signal.convolve2d(image, gaussian_x_derivative, mode = "same")
y_derivative = scipy.signal.convolve2d(image, gaussian_y_derivative, mode = "same")
x_derivative_normalize = (np.abs(x_derivative) > 0.09)
y_derivative_normalize = (np.abs(y_derivative) > 0.09)
skio.imshow(x_derivative_normalize)
skio.show()
skio.imshow(y_derivative_normalize)
skio.show()
