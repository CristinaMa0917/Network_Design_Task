import pandas as pd
import numpy as np

input = pd.read_csv('task8_test_input.csv',header=None).values.reshape(32000,20)
pre = pd.read_csv('task8_test_predict.csv',header=None).values.reshape(32000,20)
labels = []
for i in range(input):
    label = np.zeros(20)
    length = len(np.nonzero(input[i,:]))
    no_zero = reversed(input[i,:l])
    label[:l] = no_zero
    labels.append(label)

labels = np.array(input.shape)
print(np.mean(pre==labels))

