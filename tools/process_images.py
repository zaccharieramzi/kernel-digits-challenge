import numpy as np

from tools.quantization import vf_vector, gaussian_vf_vector


def process_images(n, visual_features, pins, pin_to_im,
                   gaussian_mixture=True):
    '''Vectorize the features of each images contained in the rows of X.
    '''
    if gaussian_mixture and type(visual_features) is not tuple:
        raise ValueError("if gaussian_mixture is True the second argument"
                         "should be (pi, mu, sigma)")

    if not gaussian_mixture:
        return vf_vector(pins, visual_features, pin_to_im, n)
    else:
        pi, mu, sigma = visual_features
        return gaussian_vf_vector(pins, pi, mu, sigma, pin_to_im, n)
