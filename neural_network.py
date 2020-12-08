import pandas as pd
import numpy as np
import math
import numpy as np

W_default_val = 0.1

#data = pd.read_csv('forestfires_nozeros.csv')
data = pd.read_csv('testfile.csv')
del data['X']
del data['Y']
del data['month']
del data['day']

# Inputs: FFMC DMC DC ISI temp RH wind rain	
# Output: area

weight_default = 0.01


class NeuralNetwork:

    # layers_and_units: list of units for each layer. ex) [5, 10, 20]
    def __init__(self, data, network_info, learning_rate, epoch):

        self.data = data
        self.epoch = epoch
        self.learning_rate = learning_rate

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


        # Populate unit value list
        for i in range(len(network_info)):
            layer = []
            for j in range(network_info[i]):
                layer.append(0)
            self.hidden_unit_values.append(layer)
        # Instantiate hidden unit errors here too
        self.hidden_unit_errors = self.hidden_unit_values

        # Construct random weights for hidden units
        # for each layer...
        for i in range(len(network_info)):
            layer = []
            # for each unit...
            for j in range(network_info[i]):
                unit = []

                if i == 0:
                    for k in range(8 + 1):
                        unit.append(np.random.uniform(-weight_default, weight_default))
                else:
                    for k in range(network_info[i - 1]):
                        unit.append(np.random.uniform(-weight_default, weight_default))
                
                layer.append(unit)
            self.hidden_weights.append(layer)
        
        # Construct random weights for the output unit
        for i in range(network_info[self.num_layers - 1]):
            self.output_weights.append(np.random.uniform(-weight_default, weight_default))

        # for i in range(len(self.hidden_weights)):
        #     print(f'Layer {i}')
        #     for j in range(len(self.hidden_weights[i])):
        #         print(f'Unit {j}')
        #         for k in range(len(self.hidden_weights[i][j])):
        #             print(self.hidden_weights[i][j][k])


    def propagate_input_forward(self, row):

        input_fields = [1, self.sigmoid(row['FFMC']), self.sigmoid(row['DMC']), self.sigmoid(row['DC']), 
                        self.sigmoid(row['ISI']), self.sigmoid(row['temp']), self.sigmoid(row['RH']), 
                        self.sigmoid(row['wind']), self.sigmoid(row['rain'])]
        
        # input_fields = [1, row['FFMC'], row['DMC'], row['DC'], 
        #         row['ISI'], row['temp'], row['RH'], 
        #         row['wind'], row['rain']]

        #print(input_fields)

        # for each layer...
        for i in range(len(self.hidden_weights)):
            # for each unit...
            for j in range(len(self.hidden_weights[i])):
                sum = 0

                # Sum using the input fields
                if i == 0:
                    for index in range(8 + 1):
                        sum += input_fields[index] * self.hidden_weights[i][j][index]
                    
                # Sum using the hidden units 
                else:
                    # for each weight in a unit...
                    for weight in range(len(self.hidden_weights[i][j])):
                        # sum(w * x)
                        sum += self.hidden_weights[i][j][weight] * self.hidden_unit_values[i-1][weight]
                #print(self.sigmoid(sum))
                #print(sum)
                return
                self.hidden_unit_values[i][j] = self.sigmoid(sum)

        # Compute the predicted output unit
        sum = 0
        for i in range(len(self.hidden_unit_values[self.num_layers - 1])):
            sum += self.hidden_unit_values[self.num_layers - 1][i] * self.output_weights[i]
        self.predicted_output = self.sigmoid(sum)

        # print(self.predicted_output)
        # print(self.de_sigmoid(self.predicted_output))

        return self.predicted_output



    def backpropagate_errors(self, row):

        output_error = self.predicted_output * (1 - self.predicted_output) * (self.sigmoid(row['area']) - self.predicted_output)

        # Calculate errors for the hidden units

        # Iterate layers backwards 
        for i in range(self.num_layers - 1, -1, -1):
            #for each units...
            for j in range(len(self.hidden_unit_values[i])):
                hidden_unit_error = 0 
                # The layer is the last hidden layer - update from output unit
                if i == self.num_layers - 1:
                    hidden_unit_error = self.hidden_unit_values[i][j] * (1 - self.hidden_unit_values[i][j]) * self.output_weights[j] * output_error
                # The layer isn't the last hidden layer - update from other units
                else:
                    # for each units in the next layer...
                    for k in range(len(self.hidden_unit_values[i+1])):
                        hidden_unit_error += self.hidden_weights[i+1][k][j] * self.hidden_unit_errors[i+1][k]

                    hidden_unit_error *= (self.hidden_unit_values[i][j] * (1 - self.hidden_unit_values[i][j]))
                
                self.hidden_unit_errors[i][j] = hidden_unit_error


        input_fields = [1, self.sigmoid(row['FFMC']), self.sigmoid(row['DMC']), self.sigmoid(row['DC']), 
                        self.sigmoid(row['ISI']), self.sigmoid(row['temp']), self.sigmoid(row['RH']), 
                        self.sigmoid(row['wind']), self.sigmoid(row['rain'])]

        # input_fields = [1, row['FFMC'], row['DMC'], row['DC'], 
        # row['ISI'], row['temp'], row['RH'], 
        # row['wind'], row['rain']]

        # Update the weights
        
        # for each layer..
        for i in range(self.num_layers):
            # for each unit..
            for j in range(len(self.hidden_weights[i])):
                # for each weights..
                for k in range(len(self.hidden_weights[i][j])):
                    if i == 0:
                        self.hidden_weights[i][j][k] += self.learning_rate * self.hidden_unit_errors[i][j] * input_fields[k]
                    else:
                        self.hidden_weights[i][j][k] += self.learning_rate * self.hidden_unit_errors[i][j] * self.hidden_unit_values[i-1][k]
        
    def train(self):

        num_epoch = 0

        while (num_epoch < self.epoch):

            for index, row in self.data.iterrows():
                self.propagate_input_forward(row)    
                self.backpropagate_errors(row)

            if num_epoch % 100 == 0:
                print(num_epoch)
            
            num_epoch += 1
        
        for i in range(len(self.hidden_weights)):
            print(f'Layer {i}')
            for j in range(len(self.hidden_weights[i])):
                print(f'Unit {j}')
                for k in range(len(self.hidden_weights[i][j])):
                    print(self.hidden_weights[i][j][k])

    def test(self):

        # 517 total
        correct = 0

        deviation = 10

        for index, row in self.data.iterrows():

            prediction = self.propagate_input_forward(row)

            area = row['area']

            print(f'Prediction: {self.de_sigmoid(prediction)}\tActual: {area}')
        
        #print(f'Accuracy: {correct/315.0}')
    

    def sigmoid(self, val):
        return (1 / (1 + math.exp(-val)))
    
    def de_sigmoid(self, val):
        return -math.log(val / (1-val))





neural = NeuralNetwork(data, [8], 0.05, 20000)
neural.train()
neural.test()

