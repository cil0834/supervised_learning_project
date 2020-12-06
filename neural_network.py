import pandas as pd
import numpy as np
import math
import numpy as np

W_default_val = 0.1

data = pd.read_csv('forestfires.csv')
del data['X']
del data['Y']
del data['month']
del data['day']

# Inputs: FFMC DMC DC ISI temp RH wind rain	
# Output: area



class NeuralNetwork:

    # layers_and_units: list of units for each layer. ex) [5, 10, 20]
    def __init__(self, data, network_info, learning_rate, momentum, epoch):

        self.data = data
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.num_layers = len(network_info)

        # Weights of each units of each layer
        # Search by [layer][unit][weight]
        self.hidden_weights = []

        # Result of each unit of each layer
        # Search by [layer][unit]
        self.hidden_unit_values = []

        self.hidden_unit_errors = []

        # List of weights for a single output unit
        self.output_weights = []

        # Single digit prediction of area
        self.predicted_output = 0



        # Populate unit output list
        for i in range(len(network_info)):
            layer = []
            for j in range(network_info[i]):
                layer.append(0)
            self.hidden_unit_values.append(layer)
        # Instantiate hidden unit errors here too
        self.hidden_unit_errors = self.hidden_unit_values

        # Construct small random weights to each of the weights for all hidden units in all layers
        # for each layer...
        for i in range(len(network_info)):
            layer = []
            # for each unit...
            for j in range(network_info[i]):
                unit = []

                if i == 0:
                    for k in range(8):
                        unit.append(np.random.uniform(-0.05, 0.05))
                else:
                    for k in range(network_info[i - 1]):
                        unit.append(np.random.uniform(-0.05, 0.05))
                
                layer.append(unit)
            self.hidden_weights.append(layer)
        
        # Construct small random weights to each of the weights for the output unit (area)
        for i in range(network_info[self.num_layers - 1]):
            self.output_weights.append(np.random.uniform(-0.05, 0.05))
        
        '''
        # print(f'output weights:')
        # for i in range(len(self.output_weights)):
        #     print(self.output_weights[i])

        # print('\n')

        # print(f'Hidden weights')
        # for i in range(len(self.hidden_weights)):
        #     print(f'Layer {i}')
        #     for j in range(len(self.hidden_weights[i])):
        #         print(f'Unit {j}')
        #         for k in range(len(self.hidden_weights[i][j])):
        #             print(self.hidden_weights[i][j][k])
        '''

        
    def propagate_input_forward(self, row):

        # for each layer...
        for i in range(len(self.hidden_weights)):
            # for each unit...
            for j in range(len(self.hidden_weights[i])):
                sum = 0

                # Sum using the input fields
                if i == 0:
                    x0 = row['FFMC']
                    x1 = row['DMC']
                    x2 = row['DC']
                    x3 = row['ISI']
                    x4 = row['temp']
                    x5 = row['RH']
                    x6 = row['wind']
                    x7 = row['rain']

                    sum = x0 * self.hidden_weights[i][j][0] + \
                            x1 * self.hidden_weights[i][j][1] + \
                            x2 * self.hidden_weights[i][j][2] + \
                            x3 * self.hidden_weights[i][j][3] + \
                            x4 * self.hidden_weights[i][j][4] + \
                            x5 * self.hidden_weights[i][j][5] + \
                            x6 * self.hidden_weights[i][j][6] + \
                            x7 * self.hidden_weights[i][j][7]
                # Sum using the hidden units 
                else:
                    for weight in range(len(self.hidden_weights[i][j])):
                        sum += self.hidden_weights[i][j][weight] * self.hidden_unit_values[i-1][weight]

                
                self.hidden_unit_values[i][j] = self.sigmoid(sum)

        # Compute the predicted output unit
        sum = 0
        for i in range(len(self.hidden_unit_values[self.num_layers - 1])):
            sum += self.hidden_unit_values[self.num_layers - 1][i] * self.output_weights[i]
        self.predicted_output = self.sigmoid(sum)

    def backpropagate_errors(self, row):

        output_error = self.predicted_output * (1 - self.predicted_output) * (row['area'] - self.predicted_output)


        # Construct errors for the hidden units

        # Iterate layers backwards 
        for i in range(self.num_layers - 1, -1, -1):
            # for each units...
            for j in range(len(self.hidden_unit_values[i])):
                hidden_unit_error = 0
                if i == self.num_layers - 1:
                    
                    hidden_unit_error = self.hidden_unit_values[i][j] * (1 - self.hidden_unit_values[i][j]) * self.output_weights[j] * output_error
                else:
                    for k in range(len(self.hidden_unit_values[i+1])):
                        hidden_unit_error += self.hidden_weights[i+1][k][j] * self.hidden_unit_errors[i+1][j]
                    hidden_unit_error *= self.hidden_unit_values[i][j]
                
                self.hidden_unit_errors[i][j] = hidden_unit_error
        
        # for each layer..
        for i in range(self.num_layers):
            # for each unit..
            for j in range(len(self.hidden_weights[i])):
                # for each weights..
                for k in range(len(self.hidden_weights[i][j])):
                    self.hidden_weights[i][j][k] += self.learning_rate * self.hidden_unit_errors[i][j] * self.hidden_unit_values[i][j]
        


    def train(self):

        num_epoch = 0

        while (num_epoch < self.epoch):

            for index, row in self.data.iterrows():
                self.propagate_input_forward(row)     
                self.backpropagate_errors(row)
            
            num_epoch += 1


        
        # print('\n\n')
        for i in range(len(self.hidden_unit_values)):
            print(f'Layer {i}')
            for j in range(len(self.hidden_unit_values[i])):
                print(f'{self.hidden_unit_values[i][j]}')

# print('\n\n')
# for i in range(len(self.output_weights)):
#     print(self.output_weights[i])

# print('\n')
# print(self.predicted_output)

        
        # for i in range(len(self.hidden_unit_values)):
        #     print(f'Layer {i}')
        #     for j in range(len(self.hidden_unit_values[i])):
        #         print(self.hidden_unit_values[i][j])

        





                            








        

    
    # Activation function
    def sigmoid(self, val):
        return 1 / (1 + math.e ** -val)
    
    def sigmoid_der(self, val):
        return sigmoid(val) * (1 - sigmoid(val)) 

neural = NeuralNetwork(data, [2, 2], 0.05, 0.05, 100)
neural.train()
