import pandas as pd
import numpy as np
from Decision_Tree import Regression_Tree
import random

class Random_Forest:
    def __init__(self, n_attributes, max_depth, min_values, data, x, forest_size, response):
        """

        :param n_attributes: the number of attributes taken from the total list of attributes
        :param max_depth: The max depth of the tree
        :param min_values: The minimum values allowed in a single node
        :param data: The data that is being used
        :param x: The independent variables
        :param response: The response
        """
        self.n_attributes = n_attributes
        self.forest_size = forest_size
        self.max_depth = max_depth
        self.min_values = min_values
        self.X = data[x].copy()
        self.x_columns = self.X.columns
        self.forest = []
        self.response = response
        self.Y = data[response]
        self.data = data

    def build_tree(self):
        """
        The function that builds a tree to be added to the forest
        :return:
        """
        indexes = list(range(len(self.x_columns)))
        chosen_indexes = random.sample(indexes, self.n_attributes)
        attributes = self.x_columns[chosen_indexes]
        tree = Regression_Tree(max_depth=self.max_depth, min_values=self.min_values, data=self.data, x = attributes, response=self.response)
        tree.train_tree()
        self.forest.append(tree)

    def build_forest(self):
        """
        Builds a forest based on the number of trees you want
        :return:
        """
        for i in range(self.forest_size):
            self.build_tree()

    def predict_individual(self, data_point, regression_tree):
        """
        A function that makes a prediction for a tree at a specific data point
        :param data_point: The data point being predicted
        :return: The predicted value for the data point
        """
        prediction = None
        branch = regression_tree.tree
        while prediction == None:
            category = branch['category']
            threshold = branch['threshold']

            if data_point[category] < threshold:
                branch = branch['left_branch']
            else:
                branch = branch['right_branch']
            prediction = branch.get("predicted_value")
        return(prediction)

    def predict(self, x_values):
        """
        The values that are going to be predicted
        :param x_values: A data frame of predicted values
        :return: An array of predicted values
        """
        predictions = []
        current_predictions = 0
        for x_value in x_values.iloc:
            for tree in self.forest:
                prediction = self.predict_individual(x_value, tree)
                current_predictions += prediction
            current_predictions = current_predictions / self.forest_size
            predictions.append(current_predictions)
        return predictions

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
            print("Prediction: ", prediction, " Target: ", target)
        return rss/len(predictions)


data = pd.read_csv("forestfires.csv")

Xs= ['DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

#print(data.head())
#data = data[Xs]
#print(data.head())


x_vals = data[Xs]
y_vals = data['area'].iloc
abc = Random_Forest(4, 5, 3, data, Xs, 100, 'area')

abc.build_forest()
predictions = abc.predict(x_vals)

error = abc.calculate_error(predictions, y_vals)
print(error)