import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

max_depth = 5
min_values = 10

class Regression_Tree:
    def __init__(self, max_depth, min_values, data, x, response):
        """
        A class for regression trees
        :param max_depth: The max depth of the tree
        :param min_values: The minimum values allowed in a single node
        :param data: The data that is being used
        :param x: The independent variables
        :param response: The response
        """
        self.max_depth = max_depth
        self.min_values = min_values
        self.X = data[x].copy()
        self.response = response
        self.Y = data[response]
        self.tree = {}

    def RSS(self, node):
        """
        Calculates the residual sum of squares of a tree. This function is used to determine when to split the tree
        :param node: a decision tree at a potential splitting point
        :return: The residual sum of squares of the tree
        """
        mean_val = np.mean(node)
        rss = 0
        for y_val in node:
            difference = y_val - mean_val
            difference = difference ** 2
            rss = rss + difference
        return rss

    def best_category_threshold(self, x, y):
        """
        Find the best category to split the data and then find the best value to split the value at
        :param x: The independent variables
        :param y: The response
        :return: The best category and the best threshold
        """
        # set the category and threshold to none because there is no best value for these yet.
        chosen_category = None
        chosen_threshold = None
        current_rss = np.inf

        # loop through the categories
        for category in x.columns:
            thresholds = x[category].unique().tolist()
            # loop through each threshold in the categories
            for threshold in thresholds:
                left = x[category] < threshold
                right = x[category] >= threshold
                left_rss = self.RSS(y[left])
                right_rss = self.RSS(y[right])
                rss = left_rss + right_rss
                # select a new category and threshold if a better one is found
                if rss < current_rss:
                    current_rss = rss
                    chosen_category = category
                    chosen_threshold = threshold

        return chosen_category, chosen_threshold

    def split_tree(self, x, y, depth, tree_size):
        """
        Split the tree based on learning rule
        :param x: The independent variables
        :param y: The response
        :param depth: The current depth of the tree
        :param tree_size: The current size of the tree
        :return: a decision tree or a mean if the tree is at the end
        """
        # If the tree is not big enough or if the tree is at the final depth then find the average
        if depth >= self.max_depth or tree_size <= self.min_values:
            mean = sum(y)/len(y)
            return {'predicted_value': mean}

        # We are not ready to make a prediction so we search for the best category and corresponding threshold
        else:
            # find the best category and corresponding threshold
            chosen_category, chosen_threshold  = self.best_category_threshold(x, y)
            tree = {'category': chosen_category, 'threshold': chosen_threshold}

            # find the indexes that correspond to the categories and thresholds
            left_index = x[chosen_category] < chosen_threshold
            right_index = x[chosen_category] >= chosen_threshold

            x_left = x[left_index]
            x_right = x[right_index]
            y_left = y[left_index]
            y_right = y[right_index]

            if (len(x_left) > 0):
                tree['left_branch'] = self.split_tree(x_left, y_left, depth+1, len(x_left))
            if (len(x_right > 0)):
                tree['right_branch'] = self.split_tree(x_right, y_right, depth+1, len(x_right))
            return tree

    def train_tree(self):
        self.tree = self.split_tree(self.X, self.Y, 0, len(self.X))

    def predict_individual(self, data_point):
        """
        A function that predicts and individual value
        :param data_point: The data point being predicted
        :return: The predicted value for the data point
        """
        prediction = None
        branch = self.tree
        while prediction == None:
            category = branch['category']
            threshold = branch['threshold']

            if data_point[category] < threshold:
                branch = branch['left_branch']
                if('predicted_value' in branch):
                    prediction = branch['predicted_value']
            else:
                branch = branch['right_branch']
                if('predicted_value' in branch):
                    prediction = branch['predicted_value']
        return prediction

    def predict(self, x_values):
        """
        The values that are going to be predicted
        :param x_values: A data frame of predicted values
        :return: An array of predicted values
        """
        predictions = []
        for data in x_values.iloc:
            predicted_value = self.predict_individual(data)
            predictions.append(predicted_value)
        return predictions

    def calculate_error(self, predictions, targets):
        """
        Calculates the residual sum of squares
        :param prediction: the predicted values
        :param target: the target values
        :return: Root mean squared error
        """
        rss = 0
        i = 0
        for prediction, target in zip(predictions, targets):
            i += 1
            rss += ((prediction - target) ** 2)
        return math.sqrt(rss/len(predictions))

    def predict_values(self, predictions, targets):

        for prediction, target in zip(predictions, targets):
            print("The prediction is: ", prediction, ". The true value is: ", target)

data = pd.read_csv("forestfires.csv")

data = pd.read_csv("forestfires.csv")
Xs= ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
data['area'] = np.log2(data['area'] + 1)
train, test = train_test_split(data, test_size=0.2)
train, validation = train_test_split(train, test_size=0.25)
test_y = test['area']
validation_y = validation['area']

best_error = np.inf
best_depth = 0
min_val = 10
for depth in range(1, 6):
        dt = Regression_Tree(depth, min_val, data, Xs, 'area')
        dt.train_tree()
        predictions = dt.predict(validation)
        error = dt.calculate_error(predictions, validation_y)
        print("Depth: ", depth, " Error: ", error)
        if (best_error > error):
            best_depth = depth
            best_error = error
            best_min_val = min_val

print("The best depth is: ", best_depth, " The best validation error is: ", best_error)

dt = Regression_Tree(3, 10, data, Xs, 'area')

dt.train_tree()
predictions = dt.predict(validation)
testing_error = dt.calculate_error(predictions, test_y)

print("The testing error is: ", testing_error)