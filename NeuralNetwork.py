import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Build the model
model = Sequential()
model.add(Dense(32, input_dim=100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the data
model.fit(data, labels, epochs=10, batch_size=32)
