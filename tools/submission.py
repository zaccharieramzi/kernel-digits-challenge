import os.path as op

import numpy as np
import pandas as pd


def labels_to_csv(Y, kernel="linear", algo="linear regression", user="Zac",
                  file_name=None):
    '''Saves the labels in the correct format.
        Args:
            - Y (ndarray): the labels.
            - kernel (str): the type of kernel used.
            - algo (str): the algorithm used.
            - user (str): who you are.
            - file_name (str): overwrites the file name.
    '''
    df_labels = pd.DataFrame(data=Y,
                             index=np.arange(len(Y))+1,
                             columns=["Prediction"])
    df_labels.index.names = ["Id"]
    if file_name is None:
        file_name = "Yte_{kernel}_{algo}_{user}.csv".format(
            kernel=kernel,
            algo=algo,
            user=user
        )
    outfolder_path = "submissions"
    if not os.path.isdir(outfolder_path):
        os.mkdir(outfolder_path)

    df_labels.to_csv(op.join(outfolder_path, file_name))
