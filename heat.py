import matplotlib.pyplot as plt
import numpy as np
import copy

import dataset

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

if __name__ == "__main__":
    dataset_path = "data/fuzzy_data_u1_b10/"
    x_train, y_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset(dataset_path)

    n = len(x_train)

    loss = np.array([Random(1, 4) for i range()])
