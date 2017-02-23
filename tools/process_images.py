def process_images(X, proc_type="identity", **kwargs):
    '''Vectorize the features of each images contained in the rows of X.
    '''
    if proc_type == "identity":
        return X
    elif proc_type == "bovf":
        try:
            centroids = kwargs["centroids"]
        else:
            raise AttributeError("You need centroids for bovf")
        try:
            im_to_pins = kwargs["im_to_pins"]
        else:
            raise AttributeError("You need im_to_pins for bovf")
        n = X.shape[0]
        X_proc = np.zeros((n, centroids.shape[0]))
        for i in range(n):
            pins = im_to_pins[i]
            X_proc[i, :] = vf_vector(pins, centroids)
        return X_proc
    else:
        raise ValueError("{} is not an image processing".format(proc_type))
