import random
import numpy as np

def select_indexes(n, number_of_values):
    """
    Returns an array of indexes
    :param n: length of data frame
    :return: An array of randomly selected indexes
    """
    index_array = np.linspace(0, n-1, n).tolist()

    return random.sample(index_array, number_of_values)
