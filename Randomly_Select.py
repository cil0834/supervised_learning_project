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
<<<<<<< HEAD

#sample = select_indexes(300, 200)
#print(sample)



index_array = [1, 2, 3, 4, 5]
#sample = np.linspace(0, 299, 300).tolist()

#random.sample(index_array, 3)

#print(len(select_indexes(300, 200)))

=======
>>>>>>> 84c3c5e40f93cfbbd8be83245d762a6ddb90b661
