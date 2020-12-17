from sklearn import svm
import numpy as np
import csv

# Get the x and y from a csv file without the header
def get_x_y(csv_name):
    csv_file = open(csv_name, 'r', newline = '')
    csv_file_reader = csv.reader(csv_file, delimiter = ',')
    lines = list(csv_file_reader)
    csv_file.close()

    x = []
    y = []

    for i in range(len(lines)):
        row_new = []
        
        if i == 0:
            continue
        else:
            for j in range(len(lines[i])):
                if j <= 4:
                    continue
            
                if j < len(lines[i]) - 1:
                    row_new.append(np.log(float(lines[i][j]) + 1))
                else:
                    y.append(np.log(float(lines[i][j]) + 1))
            x.append(row_new)
    
    return x, y


class SVM:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.X_train = []
        self.Y_train = []
        self.X_train, self.Y_train = get_x_y(self.train_data)

        self.test_data = test_data
        self.X_test = []
        self.Y_test = []
        self.X_test, self.Y_test  = get_x_y(self.test_data)

        #print(self.X_test)

        self.svm = svm.SVR()

    def train(self):
        self.svm = self.svm.fit(self.X_train, self.Y_train)

    def test(self):
        RMSE = 0
        N = len(self.Y_test)
        for i in range(N):
            y = self.svm.predict([self.X_test[i]])
            y_hat = self.Y_test[i]

            RMSE += ((float(y[0]) - y_hat) ** 2)

        RMSE = (RMSE / N) ** 0.5


        print(RMSE)
        return RMSE

svm = SVM('train.csv', 'test.csv')

svm.train()
svm.test()
