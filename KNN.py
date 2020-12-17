import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split


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


data = pd.read_csv("forestfires.csv")
Xs= ['temp', 'RH', 'wind', 'rain']
data['area'] = np.log2(data['area'] + 1)
train, test = train_test_split(data, test_size=0.2)
train, validation = train_test_split(train, test_size=0.25)
validation_y = validation['area']
test_y = test['area']

best_index = 0
best_error = np.inf
for i in range(1, 20):
    knn = KNN(i, Xs, 'area', train)
    predictions = knn.predict(validation)
    current_error = knn.calculate_error(predictions, validation_y)
    print("Number of neighbors: ", i, " MSE: ", current_error)
    if (current_error < best_error):
        best_error = current_error
        best_index = i

print("Best numbers of neighbors: ", best_index, " Best error: ", current_error)

for i in range(1, 20):
    knn = KNN(i, Xs, 'area', train)
    predictions = knn.predict(test)
    current_error = knn.calculate_error(predictions, test_y)
    print("Number of neighbors: ", i, " MSE Final: ", current_error)

    train, test = train_test_split(data, test_size=0.2)
    knn = KNN(best_index, Xs, 'area', train)
    predictions = knn.predict(test)
    final_error = knn.calculate_error(predictions, test_y)

    print("The error is: ", final_error)