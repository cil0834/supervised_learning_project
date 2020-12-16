import pandas as pd
import numpy as np
import math

W_default_val = 0.1

data = pd.read_csv('forestfires.csv')
del data['X']
del data['Y']
del data['month']
del data['day']

class Gradient:

    def __init__(self, data, gradient_type = 'stochastic', learning_rate = 0.00005, iteration = 100000):

        self.data = data

        FFMC = self.data['FFMC']
        FFMC = FFMC + 1
        self.data['FFMC'] = np.log(FFMC)

        print(self.data['FFMC'])

        DMC = self.data['DMC']
        DMC = DMC + 1
        self.data['DMC'] = np.log(DMC)

        DC = self.data['DC']
        DC = DC + 1
        self.data['DC'] = np.log(DC)

        ISI = self.data['ISI']
        ISI = ISI + 1
        self.data['ISI'] = np.log(ISI)

        temp = self.data['temp']
        temp = temp + 1
        self.data['temp'] = np.log(temp)

        RH = self.data['RH']
        RH = RH + 1
        self.data['RH'] = np.log(RH)

        wind = self.data['wind']
        wind = wind + 1
        self.data['wind'] = np.log(wind)

        rain = self.data['rain']
        rain = rain + 1
        self.data['rain'] = np.log(rain)

        area = self.data['area']
        area = area + 1
        self.data['area'] = np.log(area)

        self.Y = self.data['area']


        self.W0 = self.W1 = self.W2 = self.W3 = self.W4 = self.W5 = self.W6 = self.W7 = self.W8 = W_default_val

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

            # Calculate output
            output = self.calculate_output(row['FFMC'], row['DMC'], row['DC'], row['ISI'], 
                                           row['temp'], row['RH'], row['wind'], row['rain'])
            
            # Update Weight values
            self.W0 = self.W0 + self.learning_rate * (row['area'] - output) * 1
            self.W1 = self.W1 + self.learning_rate * (row['area'] - output) * row['FFMC']
            self.W2 = self.W2 + self.learning_rate * (row['area'] - output) * row['DMC']
            self.W3 = self.W3 + self.learning_rate * (row['area'] - output) * row['DC']
            self.W4 = self.W4 + self.learning_rate * (row['area'] - output) * row['ISI']
            self.W5 = self.W5 + self.learning_rate * (row['area'] - output) * row['temp']
            self.W6 = self.W6 + self.learning_rate * (row['area'] - output) * row['RH']
            self.W7 = self.W7 + self.learning_rate * (row['area'] - output) * row['wind']
            self.W8 = self.W8 + self.learning_rate * (row['area'] - output) * row['rain']

        

        

            

            

    def calculate_output(self, FFMC, DMC, DC, ISI, temp, RH, wind, rain):

        output = 1    * self.W0 + \
                 FFMC * self.W1 + \
                 DMC  * self.W2 + \
                 DC   * self.W3 + \
                 ISI  * self.W4 + \
                 temp * self.W5 + \
                 RH   * self.W6 + \
                 wind * self.W7 + \
                 rain * self.W8
         
        return output




gradient = Gradient(data = data)

output = gradient.calculate_output(85.8,	48.3,	313.4,	3.9,	18,	42,	2.7,	0) # expect 0.36

print(output)