import pandas as pd
import numpy as np
import math
import numpy as np

#data = pd.read_csv('forestfires_nozeros.csv')
data = pd.read_csv('testfile.csv')
#data = pd.read_csv('testfile_one.csv')
#data = pd.read_csv('forestfires.csv')

data_test = pd.read_csv('testfile_two.csv')



def process_dataset(dataset):
    del dataset['X']
    del dataset['Y']
    del dataset['month']
    del dataset['day']

    FFMC = dataset['FFMC']
    FFMC = FFMC + 1
    dataset['FFMC'] = np.log(FFMC)

    DMC = dataset['DMC']
    DMC = DMC + 1
    dataset['DMC'] = np.log(DMC)

    DC = dataset['DC']
    DC = DC + 1
    dataset['DC'] = np.log(DC)

    ISI = dataset['ISI']
    ISI = ISI + 1
    dataset['ISI'] = np.log(ISI)

    temp = dataset['temp']
    temp = temp + 1
    dataset['temp'] = np.log(temp)

    RH = dataset['RH']
    RH = RH + 1
    dataset['RH'] = np.log(RH)

    wind = dataset['wind']
    wind = wind + 1
    dataset['wind'] = np.log(wind)

    rain = dataset['rain']
    rain = rain + 1
    dataset['rain'] = np.log(rain)

    area = dataset['area']
    area = area + 1
    dataset['area'] = np.log(area)

    return dataset

data = process_dataset(pd.read_csv('testfile.csv'))
data_test = process_dataset(pd.read_csv('testfile_two.csv'))

data_all = process_dataset(pd.read_csv('forestfires.csv'))


# Inputs: FFMC DMC DC ISI temp RH wind rain	
# Output: area

weight_default = 0.01


class NeuralNetwork:

    # layers_and_units: list of units for each layer. ex) [5, 10, 20]
    def __init__(self, data, data_test, network_info, learning_rate, epoch):
        
        # test with this data
        self.data_test = data_test

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



    def propagate_input_forward(self, row):
        
        input_fields = [1, row['FFMC'], row['DMC'], row['DC'], 
                        row['ISI'], row['temp'], row['RH'], 
                        row['wind'], row['rain']]

        # Calculate values of each unit

        # for each layer...
        for i in range(len(self.hidden_weights)):
            # for each unit...
            for j in range(len(self.hidden_weights[i])):
                sum = 0

                # SUM(w_ji * x_ji)

                # Sum using the input fields
                if i == 0:
                    for index in range(8 + 1):
                        sum += self.hidden_weights[i][j][index] * input_fields[index]
                    
                # Sum using the hidden units 
                else:
                    # for each weight in a unit...
                    for weight in range(len(self.hidden_weights[i][j])):
                        # sum(w * x)
                        sum += self.hidden_weights[i][j][weight] * self.hidden_unit_values[i-1][weight]

                self.hidden_unit_values[i][j] = self.sigmoid(sum)


        # Compute the predicted output unit
        # No activation function needed here
        sum = 0
        for i in range(len(self.output_weights)):
            sum += self.output_weights[i] * self.hidden_unit_values[self.num_layers - 1][i]

        self.predicted_output = sum 

        return self.predicted_output



    def backpropagate_errors(self, row):

        input_fields = [1, row['FFMC'], row['DMC'], row['DC'], 
                        row['ISI'], row['temp'], row['RH'], 
                        row['wind'], row['rain']]

        # Calculate output unit error and adjust output unit weights

        output_error = row['area'] - self.predicted_output
        
        # for each weights in output
        for i in range(len(self.output_weights)):
            delta_w = self.learning_rate * output_error * self.hidden_unit_values[self.num_layers - 1][i]
            self.output_weights[i] += delta_w

        
        # Calculate hidden unit error and adjust hidden unit weights

        # for each layer in reverse order...
        for i in range(self.num_layers - 1, -1, -1):   
            # for each unit...
            for j in range(len(self.hidden_unit_values[i])):
                hidden_unit_error = 0
                if i == self.num_layers - 1:
                    hidden_unit_error = self.hidden_unit_values[i][j] * (1 - self.hidden_unit_values[i][j]) * self.output_weights[j] * output_error

                    for k in range(len(self.hidden_weights[i][j])):
                        self.hidden_weights[i][j][k] += self.learning_rate * hidden_unit_error * self.hidden_unit_values[i-1][k]

                else:
                    # for each unit in the next layer...
                    for k in range(len(self.hidden_unit_values[i+1])):
                        hidden_unit_error += self.hidden_unit_errors[i+1][k] * self.hidden_weights[i+1][k][j]
                    
                    hidden_unit_error *= (self.hidden_unit_values[i][j] * (1 - self.hidden_unit_values[i][j]))

                    for k in range(len(self.hidden_weights[i][j])):
                        if i == 0:
                            self.hidden_weights[i][j][k] += self.learning_rate * hidden_unit_error * input_fields[k]
                        else:
                            self.hidden_weights[i][j][k] += self.learning_rate * hidden_unit_error * self.hidden_unit_values[i-1][k]

                self.hidden_unit_errors[i][j] = hidden_unit_error
        
    def train(self):

        num_epoch = 0

        while (num_epoch < self.epoch):
            for index, row in self.data.iterrows():
                self.propagate_input_forward(row)
                self.backpropagate_errors(row)
            
            if num_epoch % 100 == 0:
                print(num_epoch)
            
            num_epoch += 1
        

    def test(self):

        rss_sum = 0

        for index, row in self.data_test.iterrows():
            
            prediction = self.propagate_input_forward(row)
            self.backpropagate_errors(row)

            area = row['area']
            print(f'Prediction: {prediction}\tActual: {area}')

            rss_sum += ((prediction - area) ** 2)


        print(rss_sum)
        return rss_sum


    def sigmoid(self, val):
        return (1 / (1 + math.exp(-val)))
    




neural = NeuralNetwork(data_all, data_all, [10, 8], 0.05, 10000)
neural.train()
neural.test()

