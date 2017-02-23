import numpy as np

from tools.quantization import vf_vector


def process_images(X, proc_type="identity", **kwargs):
    '''Vectorize the features of each images contained in the rows of X.
    '''
    if proc_type == "identity":
        return X
    elif proc_type == "bovf":
        try:
            centroids = kwargs["centroids"]
        except KeyError:
            raise KeyError("You need centroids for bovf")
        else:
            try:
                im_to_pins = kwargs["im_to_pins"]
            except KeyError:
                raise KeyError("You need im_to_pins for bovf")
            else:
                n = X.shape[0]
                X_proc = np.zeros((n, centroids.shape[0]))
                for i in range(n):
                    pins = im_to_pins[i]
                    X_proc[i, :] = vf_vector(pins, centroids)
                    print(i)
                return X_proc
    else:
        raise ValueError("{} is not an image processing".format(proc_type))
