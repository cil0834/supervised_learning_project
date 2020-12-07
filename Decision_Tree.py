import pandas as pd
import numpy as np

max_depth = 5
min_values = 10

data = pd.read_csv("forestfires.csv")


zyx = data.iloc[0]
y_data = data['area']
targets = y_data.tolist()
#print(zyx)
#print(zyx['month'])

#del data['X']
#del data['Y']
#del data['month']
#del data['day']

Xs= ['DC', 'ISI', 'temp', 'RH', 'wind', 'rain']



#abc = data['ISI'] < 9

#print(data[abc])
#data = data[Xs].copy()
#print(data)
#X = data.to_numpy()
#print(X)


def RSS(node):
    """
    Calculates the residual sum of squares of a tree. This function is used to determine when to split the tree
    :param tree: a decision tree at a potential splitting point
    :return: The residual sum of squares of the tree
    """
    return (np.sum(node - np.mean(node)) ** 2)

class Regression_Tree:
    def __init__(self, max_depth, min_values, data, x, response):
        """
        A class for regression trees
        :param max_depth: The max depth of the tree
        :param min_values: The minimum values allowed in a single node
        :param data: The data that is being used
        :param x: The independent variables
        :param y: The response
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

        return {'category': chosen_category, 'threshold': chosen_threshold}

    def split_tree(self, x, y, depth, tree_size):
        """
        Split the tree based on learning rule
        :param x: The independent variables
        :param y: The response
        """
        # If the tree is not big enough or if the tree is at the final depth then find the average
        if depth >= self.max_depth or tree_size <= self.min_values:
            return {'predicted_value': np.mean(y)}

        # We are not ready to make a prediction so we search for the best category and corresponding threshold
        else:
            # find the best category and corresponding threshold
            tree = self.best_category_threshold(x, y)

            # find the indes that correspond to the categories and thresholds
            chosen_category = tree['category']
            chosen_threshold = tree['threshold']
            left_index = x[chosen_category] < chosen_threshold
            right_index = x[chosen_category] >= chosen_threshold

            x_left = x[left_index]
            x_right = x[right_index]
            y_left = y[left_index]
            y_right = y[right_index]

            tree['left_branch'] = self.split_tree(x_left, y_left, depth+1, len(x_left))
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
        for data in x_values.iloc:
            predicted_value = self.predict_individual(data)
            predictions.append(predicted_value)
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
            rss += (prediction - target) ** 2

        return rss/len(predictions)

abc = Regression_Tree(2, 2, data, Xs, 'area')
abc.train_tree()
predictions = abc.predict(data)

print(abc.calculate_error(predictions, targets))