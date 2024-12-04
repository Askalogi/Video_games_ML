import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

 # Correct import

 


# Example model definition
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(91,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[RSquare()])

# Train the model (assuming X_train, y_train are defined)
history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2)



data = pd.read_csv("../Project 2 Machine Learning/vgsales.csv", index_col= "Rank")
data

# Preprosecing
columns_to_drop = ["Name", 'NA_Sales', "EU_Sales", "JP_Sales","Other_Sales"]
data.drop(columns_to_drop, axis = 1, inplace= True)
data

data['Year'] = data['Year'].fillna(data['Year'].mean())
data['Publisher'] = data['Publisher'].dropna(axis=0)
data = data.dropna(axis = 0)
data.isnull().sum() #cleaning the blank values that appeared in Publisher and Year

data ['Platform'].unique() #ok amount
data ['Genre'].unique() #ok amount
data ['Publisher'].unique() # not okay amount

pd.set_option('display.max_rows', 20)
data['Publisher'].value_counts() # theloume na doume pou arxizei na exei simasia na doume publishers
counts = data['Publisher'].value_counts()
data['Publisher'] = data['Publisher'].apply(lambda x: 'Small Publisher'if counts[x] < 50 else x )
data['Publisher'].value_counts()


onehot_columns = ['Platform','Genre','Publisher']


def onehot_endoce(data, columns):
    for column in columns:
        dummies = pd.get_dummies(data[column])
        dummies = dummies.astype(int) 
        data = pd.concat([data, dummies], axis=1)
        data.drop(column, axis=1, inplace=True)
    return data

data = onehot_endoce(data, onehot_columns)
print(data)

#SCALING
y = data['Global_Sales']
X = data.drop('Global_Sales', axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X.shape

#TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
inputs = tf.keras.Input(shape=(91,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs = inputs , outputs = outputs)

optimizer = tf.keras.optimizers.RMSprop(0.01)

model.compile(
    optimizer=optimizer,
    loss='mse'

)

batch_size = 64
epochs = 300

history = model.fit(
    X_train,
    y_train,
    validation_split = 0.2,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 0
)

#RESULTS
plt.figure(figsize= (14, 10))
epochs_range = range(1, epochs+1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs_range, train_loss, label = "Training Loss")
plt.plot(epochs_range, val_loss, label = "Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


y_pred = np.squeeze(model.predict(X_test))
result = RSquare()
result.update_state(y_test, y_pred)
print("R^2 Score:", result.result())
