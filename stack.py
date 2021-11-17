import numpy as np
import scipy
from scipy import stats
import skimage as sk
import skimage.io as skio
import os
from skimage.transform import resize
import scipy
import cv2

def gaussian_stack_check(image, num_layers):
    """
    This is a function that checks if *gaussian_stack()* is implemented correctly.
    Implementation utilizes *scipy.ndimage.gaussian_filter()*.
    None of the following parts actually uses the function to generate image.
    Please ignore the whole implementation during grading.
    """
    result = [image]
    for i in range(num_layers):
        image_buffer = np.dstack(
            [scipy.ndimage.gaussian_filter(image[:, :, 0], 2 * (i + 1), truncate = 2),
            scipy.ndimage.gaussian_filter(image[:, :, 1], 2 * (i + 1), truncate = 2),
            scipy.ndimage.gaussian_filter(image[:, :, 2], 2 * (i + 1), truncate = 2)])
        image_buffer = np.dstack(
            [(image_buffer[:, :, 0] -  np.min(image_buffer[:, :, 0]))/ \
            (np.max(image_buffer[:, :, 0]) - np.min(image_buffer[:, :, 0])),
            (image_buffer[:, :, 1] -  np.min(image_buffer[:, :, 1]))/ \
            (np.max(image_buffer[:, :, 1]) - np.min(image_buffer[:, :, 1])),
            (image_buffer[:, :, 2] -  np.min(image_buffer[:, :, 2]))/ \
            (np.max(image_buffer[:, :, 2]) - np.min(image_buffer[:, :, 2]))
            ])
        result.append(image_buffer)
    return result

def gaussian_stack(image, num_layers, factor):
    """
    Finds the gaussian stack of a given image.
    :param image: input image from which gaussian stack is computed.
    :param num_layers: number of layers in the gaussian stack.
    :param factor: the parameter that decides the size of Gaussian filter at
        each level: factor * (i + 1), where i is the level number.
    """
    result = [image]
    for i in range(num_layers):
        gaussian_filter_1d = cv2.getGaussianKernel(ksize = factor *(i+1)+1,
            sigma = factor // 2 * (i+1))
        gaussian_filter = np.dot(gaussian_filter_1d, gaussian_filter_1d.T)
        if len(image.shape) == 2:
            image_buffer = scipy.signal.convolve2d(image, gaussian_filter,
                mode = "same")
        else:
            image_buffer = np.dstack(
                [scipy.signal.convolve2d(image[:, :, 0], gaussian_filter,
                mode = "same"),
                scipy.signal.convolve2d(image[:, :, 1], gaussian_filter,
                mode = "same"),
                scipy.signal.convolve2d(image[:, :, 2], gaussian_filter,
                mode = "same")])
        result.append(image_buffer)
    return result

def laplacian_stack(image, num_layers, factor):
    """
    Finds the laplacian stack of a given image.
    :param image: input image from which laplacian stack is computed.
    :param num_layers: number of layers in the laplacian stack.
    :param factor: the parameter that decides the size of Gaussian filter at
        each level: factor * (i + 1), where i is the level number.
    """
    gaussian_blur = gaussian_stack(image, num_layers, factor)
    result = []
    for i in range(0, len(gaussian_blur) - 1):
        high_res = gaussian_blur[i] - gaussian_blur[i + 1]
        result.append(high_res)
    result.append(gaussian_blur[-1])
    return result

def main():
    image = skio.imread("oraple.jpeg")
    image = sk.img_as_float(image)
    skio.imshow(image)
    skio.show()
    # Calculate the gaussian stack of "oraple.jpeg".
    gaussian_result = gaussian_stack(image, 5, 6)
    # Normalize values in the array to [0, 1].
    for image_buffer in gaussian_result:
        if len(image_buffer.shape) == 2:
            image_buffer = (image_buffer -  np.min(image_buffer))/ \
            (np.max(image_buffer) - np.min(image_buffer))
        else:
            image_buffer = np.dstack(
                [(image_buffer[:, :, 0] -  np.min(image_buffer[:, :, 0]))/ \
                (np.max(image_buffer[:, :, 0]) - np.min(image_buffer[:, :, 0])),
                (image_buffer[:, :, 1] -  np.min(image_buffer[:, :, 1]))/ \
                (np.max(image_buffer[:, :, 1]) - np.min(image_buffer[:, :, 1])),
                (image_buffer[:, :, 2] -  np.min(image_buffer[:, :, 2]))/ \
                (np.max(image_buffer[:, :, 2]) - np.min(image_buffer[:, :, 2]))
                ])
        skio.imshow(image_buffer)
        skio.show()
    # Calculate the gaussian stack of "oraple.jpeg".
    laplacian_result = laplacian_stack(image, 6, 6)
    # Normalize values in the array to [0, 1].
    for image_buffer in laplacian_result:
        if len(image_buffer.shape) == 2:
            image_buffer = (image_buffer -  np.min(image_buffer))/ \
            (np.max(image_buffer) - np.min(image_buffer))
        else:
            image_buffer = np.dstack(
                [(image_buffer[:, :, 0] -  np.min(image_buffer[:, :, 0]))/ \
                (np.max(image_buffer[:, :, 0]) - np.min(image_buffer[:, :, 0])),
                (image_buffer[:, :, 1] -  np.min(image_buffer[:, :, 1]))/ \
                (np.max(image_buffer[:, :, 1]) - np.min(image_buffer[:, :, 1])),
                (image_buffer[:, :, 2] -  np.min(image_buffer[:, :, 2]))/ \
                (np.max(image_buffer[:, :, 2]) - np.min(image_buffer[:, :, 2]))
                ])
        skio.imshow(image_buffer)
        skio.show()

if __name__ == "__main__":
    main()
