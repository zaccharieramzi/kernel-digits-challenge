T_tr = np.loadtxt("Ytr.csv", skiprows=1, usecols=(1,), delimiter=',')

def load_images(type="train"):
    '''
    Args :
           - type (str): "train" or "test"
    Returns :
           - X ndarray (5000,3072)
    '''
