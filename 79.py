import numpy as np
import pandas as pd

PINES = pd.read_csv("S1_Data.csv")

x = np.load('X.npy')

x_tr = []
y_tr = []
x_te = []
y_te = []
for i in range(len(x)):
	if (len(x[i]) == 79) and (PINES['Holdout'][i] == 'Train'):
		x_tr.append(x[i])
		y_tr.append(PINES['Rating'][i])
	if (len(x[i]) == 79) and (PINES['Holdout'][i] == 'Test'):
		x_te.append(x[i])
		y_te.append(PINES['Rating'][i])

x_train = np.array(x_tr)
y_train = np.array(y_tr)
x_test = np.array(x_te)
y_test = np.array(y_te)

np.save('x_train_79.npy', x_train)
np.save('x_test_79.npy', x_test)
np.save('y_train_79.npy', y_train)
np.save('y_test_79.npy', y_test)
