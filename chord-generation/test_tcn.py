import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from random import shuffle

from tcn import TCN

import pickle

dataset = pickle.load(open('output_sequences.p', 'rb'))

lookback_window = 4
TOT_EPOCHS = 400
SPLIT = 0.7

shuffle(dataset)

x, y = [], []
for j in range(len(dataset)):
    for i in range(lookback_window, len(dataset[j])):
        x.append(dataset[j][i - lookback_window:i])
        y.append(dataset[j][i])
x = np.array(x)
y = np.array(y)

X = np.reshape(x, (x.shape[0], lookback_window, 1))
Y = np.reshape(y, (y.shape[0], 1))

print(X.shape)
print(Y.shape)

i = Input(shape=(lookback_window, 1))
m = TCN(kernel_size=4, dilations=[1, 2, 4], dropout_rate=0.25, use_skip_connections=True)(i)
m = Dense(25, activation='softmax')(m)
model = Model(inputs=[i], outputs=[m])
model.summary()
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['acc', 'mae'])

print('Train...')
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

dataset_len = len(X)
X_train, Y_train = X[:int(dataset_len*SPLIT)], Y[:int(dataset_len*SPLIT)]
X_test, Y_test = X[int(dataset_len*SPLIT):], Y[int(dataset_len*SPLIT):]

for epoch in range(TOT_EPOCHS):
    X_train_s = []
    Y_train_s = []
    index_shuf = list(range(len(X_train)))
    shuffle(index_shuf)
    for i in index_shuf:
        X_train_s.append(X_train[i])
        Y_train_s.append(Y_train[i])
    X_train = np.array(X_train_s)
    Y_train = np.array(Y_train_s)

    print(f'EPOCH {epoch+1}/{TOT_EPOCHS}')
    stats = model.fit(X_train, Y_train, verbose=1)
    train_loss_list.append(stats.history['loss'][0])
    train_acc_list.append(stats.history['acc'][0])

    # testing
    X_test_s = []
    Y_test_s = []
    index_shuf = list(range(len(X_test)))
    shuffle(index_shuf)
    for i in index_shuf:
        X_test_s.append(X_test[i])
        Y_test_s.append(Y_test[i])
    X_test = np.array(X_test_s)
    Y_test = np.array(Y_test_s)

    s = model.evaluate(X_test, Y_test, verbose=1)
    test_loss_list.append(s[0])
    test_acc_list.append(s[1])

plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.legend(['train loss', 'test loss'])
plt.show()

plt.plot(train_acc_list)
plt.plot(test_acc_list)
plt.legend(['train accuracy', 'test accuracy'])
plt.show()

prediction = np.array([7, 2, 21, 0, 7, 2, 21, 0])
for i in range(16):
    pred = model.predict(np.reshape(prediction[i: i+lookback_window], (1, lookback_window, 1)))[0]
    index = np.where(pred == np.amax(pred))[0]
    prediction = np.append(prediction, index)

print(prediction)
print()

CHORDS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B', 'Cm', 'C#m', 'Dm', 'Ebm', 'Em', 'Fm', 'F#m', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm', 'No Chord']
for chord in prediction:
    print(CHORDS[chord], end = ' ')
print()
