import pandas as pd
import numpy as np
import math

W_default_val = 1

class Gradient:

    def __init__(self, file, gradient_type = 'stochastic', learning_rate = 0.000001, iteration = 10000):

        data_unprocessed = pd.read_csv(file)

        self.Y = data_unprocessed['area']

        #print(self.Y[0])

        # Process data
        del data_unprocessed['X']
        del data_unprocessed['Y']
        del data_unprocessed['month']
        del data_unprocessed['day']
        #del data_unprocessed['area']

        self.data = data_unprocessed
        self.learning_rate = learning_rate
        row, self.col = self.data.shape

        self.W = []
        for i in range(self.col + 1 - 1):
            self.W.append(W_default_val)

        for i in range(iteration):

            if i % 100 == 0:
                print(i)
            self.train()


    def train(self):
        
        # Iterate through all training examples
        for index, row in self.data.iterrows():

           

            # if index < 140:
            #     continue

            # if index > 143:
            #     break

            #print('Example ' + str(index))

            # Calculate output
            output = self.calculate_output(row['FFMC'], row['DMC'], row['DC'], row['ISI'], 
                                           row['temp'], row['RH'], row['wind'], row['rain'])
            #print(str(self.Y[index]) + ',' + str(output))
            #print(str(self.Y[index]) + ', ' + str(output) + ', ' + str(self.Y[index] - output))
            
            # Update Weight values
            self.W[0] = self.W[0] + self.learning_rate * (row['area'] - output) * 1
            self.W[1] = self.W[1] + self.learning_rate * (row['area'] - output) * row['FFMC']
            self.W[2] = self.W[2] + self.learning_rate * (row['area'] - output) * row['DMC']
            self.W[3] = self.W[3] + self.learning_rate * (row['area'] - output) * row['DC']
            self.W[4] = self.W[4] + self.learning_rate * (row['area'] - output) * row['ISI']
            self.W[5] = self.W[5] + self.learning_rate * (row['area'] - output) * row['temp']
            self.W[6] = self.W[6] + self.learning_rate * (row['area'] - output) * row['RH']
            self.W[7] = self.W[7] + self.learning_rate * (row['area'] - output) * row['wind']
            self.W[8] = self.W[8] + self.learning_rate * (row['area'] - output) * row['rain']

            # for i in range(len(self.W)):
            #     print(self.W[i])
            

            

    def calculate_output(self, FFMC, DMC, DC, ISI, temp, RH, wind, rain):

        output = 1    * self.W[0] + \
                 FFMC * self.W[1] + \
                 DMC  * self.W[2] + \
                 DC   * self.W[3] + \
                 ISI  * self.W[4] + \
                 temp * self.W[5] + \
                 RH   * self.W[6] + \
                 wind * self.W[7] + \
                 rain * self.W[8]
         
        return output




gradient = Gradient('forestfires.csv')

output = gradient.calculate_output(85.8,	48.3,	313.4,	3.9,	18,	42,	2.7,	0)

print(output)