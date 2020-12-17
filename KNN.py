import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



class KNN:
    def __init__(self, n_neighbors, x, response, data):
        """
        The initialization of k nearest neighbors method
        :param n_neighbors: The number of neighbors that will be looked at
        :param x: the x variables
        :param response: the response variable
        :param data: the data frame
        """

        self.X = data[x].copy()
        self.x_columns = self.X.columns
        self.response = response
        self.Y = data[response]
        self.data = data
        self.n_neighbors = n_neighbors

    def calculate_distance(self, data_point):
        """
        Calculates the euclidean distance of the n nearest neighbors of the point we want to predict
        :param data_point: the point we are going to predict
        :return: An array of the responses of the closest neighbors
        """
        distances = []
        responses = []
        index = 0
        for response, training_example in zip(self.Y.iloc, self.X.iloc):
            distance = 0
            for attribute in self.x_columns:
                distance += (data_point[attribute] - training_example[attribute]) ** 2

            if (len(distances) < self.n_neighbors):
                distances.append(distance)
                responses.append(response)

            elif (len(distances) == self.n_neighbors):
                largest_distance = 0
                largest_index = 0
                for i in range(len(distances)):
                    if(largest_distance < distances[i]):
                        largest_distance = distances[i]
                        largest_index = i

                if (largest_distance > distance):
                    distances[largest_index] = distance
                    responses[largest_index] = response
            index = index + 1
        return responses

    def make_prediction(self, testing_example):
        """
        Makes a single prediction for a data point
        :param testing_example: A single testing examples
        :return: The prediciton for the data point
        """
        responses = self.calculate_distance(testing_example)

        # average the responses and return the prediction
        return (sum(responses)/len(responses))


    def predict(self, testing_examples):
        """
        Makes predictions on all of the testing examples
        :param testing_examples: An array of testing examples
        :return: a prediction on all of the testing examples
        """
        predictions = []
        for testing_example in testing_examples.iloc:
            prediction = self.make_prediction(testing_example)
            predictions.append(prediction)
        return predictions

    def calculate_error(self, predictions, targets):
        """
        Calculates the RSS of the predictions
        :param predictions: An array of predictions
        :param targets: An array of the true values
        :return: RSS of the predictions
        """
        rss = 0
        for prediction, target in zip(predictions, targets):
            rss += ((prediction - target) ** 2)
            print("Prediction: ", prediction, " Target: ", target)
        return rss / len(predictions)

    def calculate_error(self, predictions, targets):
        """
        Calculates the residual sum of squares
        :param prediction: the predicted values
        :param target: the target values
        :return: Residual sum of squares
        """
        rss = 0
        for prediction, target in zip(predictions, targets):
            rss += ((prediction - target) ** 2)
            #print("Prediction: ", prediction, " Target: ", target, " RSS: ", rss)
        return math.sqrt(rss/len(predictions))


Xs= ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
train = pd.read_csv('train.csv')
validation = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')

train['area'] = np.log2(train['area'] + 1)
validation['area'] = np.log2(validation['area'] + 1)
test['area'] = np.log2(test['area'] + 1)
validation_y = validation['area']
test_y = test['area']
'''
best_index = 0
best_error = np.inf
errors = []
for i in range(1, 21):
    knn = KNN(i, Xs, 'area', train)
    predictions = knn.predict(validation)
    current_error = knn.calculate_error(predictions, validation_y)
    errors.append(current_error)
    print("Number of neighbors: ", i, " MSE: ", current_error)
    if (current_error < best_error):
        best_error = current_error
        best_index = i

print("Best numbers of neighbors: ", best_index, " Best error: ", best_error)

y = np.linspace(1, 20, 20)
plt.plot(errors)
plt.scatter(y, errors, marker='o');
plt.ylabel('Validation RMSE')
plt.xlabel("Number of Neighbors")
plt.title("RMSE vs. Number of Neighbors")
plt.show()
'''

training = [train, validation]
training = pd.concat(training)

knn = KNN(14, Xs, 'area', training)
predictions = knn.predict(test)
current_error = knn.calculate_error(predictions, test_y)

print(current_error)

