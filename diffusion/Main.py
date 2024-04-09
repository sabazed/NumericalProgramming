import matplotlib
import matplotlib.pyplot as plt

import Adams
import Kutta
from utils import ImageUtils, MathUtils
from utils.Benchmark import Benchmark

matplotlib.use('TkAgg')


def plot(img, img_noise, adams, kutta):
    _, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0][0].imshow(img, cmap='gray')
    axes[0][0].set_title('Original')
    axes[0][0].axis('on')

    axes[0][1].imshow(img_noise, cmap='gray')
    axes[0][1].set_title('Noise')
    axes[0][1].axis('on')

    axes[1][0].imshow(adams, cmap='gray')
    axes[1][0].set_title(Adams.REPR)
    axes[1][0].axis('on')

    axes[1][1].imshow(kutta, cmap='gray')
    axes[1][1].set_title(Kutta.REPR)
    axes[1][1].axis('on')

    plt.show()


def adams(img):
    img_denoise = Adams.Adams(img, 0.1_69, 1, 0.1_420, 1).apply()
    print("Trunc Error:", MathUtils.get_trunc_error(img, img_denoise))
    return img_denoise


def kutta(img):
    img_denoise = Kutta.Kutta(img, 0.1_69, 1, 0.1_420, 1).apply()
    print("Trunc Error:", MathUtils.get_trunc_error(img, img_denoise))
    return img_denoise

if __name__ == '__main__':

    url = 'imgs/3.png'
    img = ImageUtils.get_image(url)
    # We fetch an Image, grayscale it and add some noise using Gaussian one
    # ImageUtils class contains some useful functions for working with an image
    img_gray = ImageUtils.apply_grayscale(img)
    img_gray_normal = ImageUtils.normalize_image(img_gray)
    img_noise = ImageUtils.apply_noise(img_gray_normal)
    # I use the benchmark object to calculate time durations for each method
    benchmark = Benchmark()

    # First we try Adam
    benchmark.start_bench(Adams.REPR)
    img_adams = adams(img_noise)
    benchmark.end_bench(Adams.REPR)

    # Then we try Kutta
    benchmark.start_bench(Kutta.REPR)
    img_kutta = kutta(img_noise)
    benchmark.end_bench(Kutta.REPR)

    ### Comparison Criteria ###
    # >>> Error
    # > I calculate difference between noisy and denoised images as truncation error
    # > Target results are > 2, however error gets smaller as we decrease diffusion coefficient and increase lambda
    # >>> Smoothness
    # > I was unable to udentify and uniquely ultimate numbers for my parameters, as most of them needed to be altered
    #   depending on different scenarios, however both methods give very similar results with same arguments passed
    # >>> Duration
    # > Adam's method tends to be somewhat slower mostly as the evaluation needs much more resources, so in most of the
    #   cases using Kutta was a better option, but still not very consistent behaviour.

    # Finally, we plot the results
    plot(img_gray_normal, img_noise, img_adams, img_kutta)


