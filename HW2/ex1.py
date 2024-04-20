import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, cv2.IMREAD_COLOR)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
  # Need to implement here

    # Get image dimensions
    rows, cols, channels = img.shape

    # Calculate padding size to maintain original size after filtering
    pad_size = filter_size // 2

    # Create padded image with new dimensions
    padded_img = np.zeros((rows + 2 * pad_size, cols + 2 * pad_size, channels), dtype=img.dtype)

    print(padded_img[pad_size:-pad_size:, :pad_size, :].shape)

    # Replicate padding - fill borders with mirrored values
    for channel in range(channels):
        padded_img[:pad_size, pad_size:-pad_size, channel] = img[0, :, channel]  # Top
        padded_img[-pad_size:, pad_size:-pad_size:, channel] = img[rows - 1, :, channel]  # Bottom
        padded_img[pad_size:-pad_size, :pad_size, channel] = img[:, 0, channel][0]  # Left
        padded_img[pad_size:-pad_size, -pad_size:, channel] = img[:, cols - 1, channel][0]  # Right

    # Fill center with original image
    padded_img[pad_size:-pad_size, pad_size:-pad_size, :] = img

    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    padded_img = padding_img(img, filter_size)

    # Initialize result image
    smoothed_img = np.zeros_like(img)

    # Iterate through each pixel of the original image (within non-padded region)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract filter window from padded image
            window = padded_img[i:i + filter_size, j:j + filter_size, :]

            # Apply mean filter (average all values in window)
            average = np.mean(window, axis=(0, 1))

            # Set corresponding pixel in smoothed image
            smoothed_img[i, j, :] = average

    return smoothed_img
  # Need to implement here


def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    # Need to implement here
    padded_img = padding_img(img, filter_size)

    # Initialize result image
    smoothed_img = np.zeros_like(img)

    # Iterate through each pixel of the original image (within non-padded region)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract filter window from padded image
            window = padded_img[i:i + filter_size, j:j + filter_size, :]

            # Flatten the window for easier median calculation
            flat_window = window.flatten()

            # Apply median filter (find the median value in the window)
            median = np.median(flat_window)

            # Set corresponding pixel in smoothed image
            smoothed_img[i, j, :] = median

    return smoothed_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here
    # Check if images have the same dimensions
    if gt_img.shape != smooth_img.shape:
        raise ValueError("Ground truth and smoothed images must have the same dimensions.")

    # Convert images to float for calculations (assuming they are uint8)
    gt_img = gt_img.astype(np.float32)
    smooth_img = smooth_img.astype(np.float32)

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((gt_img - smooth_img) ** 2)

    # If the MSE is zero, avoid division by zero (PSNR is infinite)
    if mse == 0:
        return float('inf')

    # Maximum possible pixel value (assuming images are uint8)
    max_pixel = 255.0

    # Calculate PSNR (in dB)
    psnr_score = 10 * np.log10(max_pixel ** 2 / mse)

    return psnr_score


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = r"ex1_images\noise.png" # <- need to specify the path to the noise image
    img_gt = r"ex1_images\ori_img.png" # <- need to specify the path to the gt image

    #print(cv2.imread(img_noise, cv2.IMREAD_COLOR).shape)
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

    