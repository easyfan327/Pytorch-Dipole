import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    folder = "Bi-AttnConcat"
    test_auc = np.zeros([30, 1])
    test_auc_idx = np.zeros([30, 1])

    for i in range(30):
        file = r"{}.csv".format(i)
        path = os.path.join(folder, file)

        df = pd.read_csv(path, header=None)
        max_test_auc = df[3].max()
        max_test_auc_idx = df[3].idxmax()
        test_auc[i] = max_test_auc
        test_auc_idx[i] = max_test_auc_idx

    print(test_auc.mean())
    print(test_auc_idx.mean())

