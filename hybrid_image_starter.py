import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import scipy
import skimage as sk
import skimage.io as skio
from skimage import color
import os
from skimage.transform import resize
import scipy
import cv2

def hybrid_image_bw(image1, image2, sigma1, sigma2, weights):
    """
    Hybrid image1 and image2 together. Viewing from near, one would see image2,
    while viewing from afar, one would see image1.
    The resulting image is gray-scale.
    :param image1: input image 1. Gray-scale.
    :param image2: input image 2. Gray-scale.
    :param sigma1: standard deviation of the Gaussian kernel applied to image 1.
    :param sigma2: standard deviation of the Gaussian kernel applied to image 2.
    :weights: relative weights applied to the two processed images while adding
        them together.
    """
    gaussian_kernel_im1 = cv2.getGaussianKernel(
        ksize = np.ceil(sigma1 * 2).astype(np.int), sigma = sigma1)
    gaussian_kernel_im1 = np.dot(gaussian_kernel_im1, gaussian_kernel_im1.T)
    gaussian_kernel_im2 = cv2.getGaussianKernel(
        ksize = np.ceil(sigma2 * 2).astype(np.int), sigma = sigma2)
    gaussian_kernel_im2 = np.dot(gaussian_kernel_im2, gaussian_kernel_im2.T)
    image1_blur = scipy.signal.convolve2d(image1, gaussian_kernel_im1, mode = "same")
    image2_blur = scipy.signal.convolve2d(image2, gaussian_kernel_im2, mode = "same")
    image2_hf = image2 - image2_blur
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image1_blur)))))
    plt.show()
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image2_hf)))))
    plt.show()
    return image1_blur * weights[0] + image2_hf * weights[1]

def hybrid_image_color(image1, image2, sigma1, sigma2, weights):
    """
    Hybrid image1 and image2 together. Viewing from near, one would see image2,
    while viewing from afar, one would see image1.
    The resulting image is colored.
    :param image1: input image 1. Colored.
    :param image2: input image 2. Colored.
    :param sigma1: standard deviation of the Gaussian kernel applied to image 1.
    :param sigma2: standard deviation of the Gaussian kernel applied to image 2.
    :weights: relative weights applied to the two processed images while adding
        them together.
    """
    gaussian_kernel_im1 = cv2.getGaussianKernel(
        ksize = np.ceil(sigma1 * 2).astype(np.int), sigma = sigma1)
    gaussian_kernel_im1 = np.dot(gaussian_kernel_im1, gaussian_kernel_im1.T)
    gaussian_kernel_im2 = cv2.getGaussianKernel(
        ksize = np.ceil(sigma2 * 2).astype(np.int), sigma = sigma2)
    gaussian_kernel_im2 = np.dot(gaussian_kernel_im2, gaussian_kernel_im2.T)
    # Blur each channel of image 1 and image 2, and stack them together to get
    # the new image.
    image1_blur = np.dstack(
        [scipy.signal.convolve2d(image1[:, :, 0], gaussian_kernel_im1, mode = "same"),
        scipy.signal.convolve2d(image1[:, :, 1], gaussian_kernel_im1, mode = "same"),
        scipy.signal.convolve2d(image1[:, :, 2], gaussian_kernel_im1, mode = "same")
        ])
    image2_blur = np.dstack(
        [scipy.signal.convolve2d(image2[:, :, 0], gaussian_kernel_im1, mode = "same"),
        scipy.signal.convolve2d(image2[:, :, 1], gaussian_kernel_im1, mode = "same"),
        scipy.signal.convolve2d(image2[:, :, 2], gaussian_kernel_im1, mode = "same")
        ])
    image2_hf = image2 - image2_blur
    return image1_blur * weights[0] + image2_hf * weights[1]

def hybrid_cat():
    im1 = plt.imread('./DerekPicture.jpg')/255.
    im2 = plt.imread('./nutmeg.jpg')/255

    # Next align images (this code is provided, but may be improved)
    im2_aligned, im1_aligned = align_images(im2, im1)

    im1_aligned_gray = 0.2989 * im1_aligned[:, :, 0] + 0.5870 * im1_aligned[:, :, 1] + \
        0.1140 * im1_aligned[:, :, 2]
    im2_aligned_gray = 0.2989 * im2_aligned[:, :, 0] + 0.5870 * im2_aligned[:, :, 1] + \
        0.1140 * im2_aligned[:, :, 2]
    hybrid_bw = hybrid_image_bw(im1_aligned_gray, im2_aligned_gray, 12, 17, [1, 2])
    hybrid_bw = hybrid_bw[300: 1100, :]
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid_bw)))))
    plt.show()
    plt.imshow(hybrid_bw, cmap = "gray")
    plt.show()

    hybrid_color = hybrid_image_color(im1_aligned, im2_aligned, 12, 8, [1, 5])
    # Normalize the resulting hybrid image to make every value in the array fall
    # in [0, 1]. 
    hybrid_color = np.dstack(
        [(hybrid_color[:, :, 0] -  np.min(hybrid_color[:, :, 0]))/ \
        (np.max(hybrid_color[:, :, 0]) - np.min(hybrid_color[:, :, 0])),
        (hybrid_color[:, :, 1] -  np.min(hybrid_color[:, :, 1]))/ \
        (np.max(hybrid_color[:, :, 1]) - np.min(hybrid_color[:, :, 1])),
        (hybrid_color[:, :, 2] -  np.min(hybrid_color[:, :, 2]))/ \
        (np.max(hybrid_color[:, :, 2]) - np.min(hybrid_color[:, :, 2]))
        ])
    plt.imshow(hybrid_color)
    plt.show()

def hybrid_efros():
    im1 = plt.imread('efros.jpeg')/255.
    im2 = plt.imread('angjoo.jpeg')/255
    im1 = np.dstack([im1, im1, im1])

    im2_aligned, im1_aligned = align_images(im2, im1)
    im1_aligned_gray = 0.2989 * im1_aligned[:, :, 0] + \
        0.5870 * im1_aligned[:, :, 1] + \
        0.1140 * im1_aligned[:, :, 2]
    im2_aligned_gray = 0.2989 * im2_aligned[:, :, 0] + \
        0.5870 * im2_aligned[:, :, 1] + \
        0.1140 * im2_aligned[:, :, 2]
    hybrid = hybrid_image_bw(im1_aligned_gray, im2_aligned_gray, 10, 8, [8, 42])
    hybrid = hybrid[550:]
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))))
    plt.show()
    plt.imshow(hybrid, cmap = "gray")
    plt.show()

def hybrid_wolf():
    im1 = plt.imread('lion.jpeg')/255.
    im2 = plt.imread('wolf.jpeg')/255

    im1_aligned, im2_aligned = align_images(im1, im2)
    im1_aligned_gray = 0.2989 * im1_aligned[:, :, 0] + 0.5870 * im1_aligned[:, :, 1] + \
        0.1140 * im1_aligned[:, :, 2]
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned_gray)))))
    plt.show()
    im2_aligned_gray = 0.2989 * im2_aligned[:, :, 0] + 0.5870 * im2_aligned[:, :, 1] + \
        0.1140 * im2_aligned[:, :, 2]
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned_gray)))))
    plt.show()
    hybrid = hybrid_image_bw(im2_aligned_gray, im1_aligned_gray, 6, 4, [1, 1])
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))))
    plt.show()
    plt.imshow(hybrid, cmap = "gray")
    plt.show()

if __name__ == "__main__":
    hybrid_cat()
    hybrid_efros()
    hybrid_wolf()
