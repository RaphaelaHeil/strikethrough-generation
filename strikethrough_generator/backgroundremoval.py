"""
Code to remove background noise from a handwritten word image. Original matlab code by Anders Hast
(anders.hast@it.uu.se), adapted to Python by Raphaela Heil (raphaela.heil@it.uu.se).

See also: P. Singh, E. Vats and A. Hast, "Learning Surrogate Models of Document Image Quality Metrics for Automated
Document Image Processing," 2018 13th IAPR International Workshop on Document Analysis Systems (DAS), 2018, pp. 67-72,
doi: 10.1109/DAS.2018.14.
"""

from math import ceil

import numpy as np
from scipy import signal
from skimage import filters


def __calculateGaussianKernel(width=5, sigma=1.):
    ax = np.arange(-width // 2 + 1., width // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)


def __calculateMaskParameters(arraySize, sz):
    if type(arraySize) == tuple:
        if sz == 0:
            kernelSize = ceil(max(arraySize))
        else:
            scale = min(arraySize) / sz
            width = ceil(arraySize[0] / scale)
            height = ceil(arraySize[1] / scale)
            kernelSize = ceil(max((width, height)))

        if kernelSize % 2 == 0:
            kernelSize = kernelSize + 1

        sigma = kernelSize / 6.0
    else:
        if sz == 0:
            kernelSize = ceil(arraySize)
        else:
            scale = arraySize / sz
            kernelSize = ceil(arraySize / scale)

        if kernelSize % 2 == 0:
            kernelSize = kernelSize + 1

        sigma = kernelSize / 6.0

    return kernelSize, sigma


def __applyFilters(image: np.ndarray, so, sz: int) -> np.ndarray:
    imageShape = image.shape
    N, sigma = __calculateMaskParameters(so, sz)
    kernel = __calculateGaussianKernel(N, sigma)
    divisor = signal.fftconvolve(np.ones(imageShape).astype('float'), kernel, 'same')
    numerator = signal.fftconvolve(image.astype('float'), kernel, 'same')
    filteredImage = np.divide(numerator, divisor)
    return filteredImage


def __blurryBandpassFilter(image: np.ndarray, blurringMaskSize: int, threshold: float) -> np.ndarray:
    if blurringMaskSize > 1:
        thickMask = __applyFilters(image, blurringMaskSize, 0)
    else:
        thickMask = image

    p2 = __applyFilters(image, image.shape, 300)
    im2 = p2 - thickMask

    th2 = filters.threshold_otsu(p2 - thickMask)
    thresholdedImage = im2 > (th2 * threshold)
    return thresholdedImage


def __thinBandpassFilter(image, noiseMaskSize, enhanceContrast) -> np.ndarray:
    if noiseMaskSize > 1:
        thinMask = __applyFilters(image, noiseMaskSize, 0)
    else:
        thinMask = image

    p2 = __applyFilters(image, image.shape, 100)
    im2 = p2 - thinMask

    nim2 = np.zeros(im2.shape)
    nim2[im2 > 0] = im2[im2 > 0]

    if enhanceContrast:
        nim2 = nim2 - nim2.min()
        nim2 = nim2 / nim2.max()

    return nim2


def backgroundRemoval(image: np.ndarray, blurringMaskSize: int, noiseMaskSize: int, threshold: float,
                      enhanceContrast: bool) -> np.ndarray:
    nim1 = __blurryBandpassFilter(image, blurringMaskSize, threshold)
    nim2 = __thinBandpassFilter(image, noiseMaskSize, enhanceContrast)

    if enhanceContrast:
        nim2 = nim2 - nim2.min()
        nim2 = nim2 / nim2.max()

    result = 255 - (nim1 * nim2)

    return result
