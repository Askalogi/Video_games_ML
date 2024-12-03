import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# reading and cleaning the data (already cleaned)
music_data = pd.read_csv("../Project 2 Machine Learning/music.csv")

#splitting the data (input and output)
X = music_data.drop(columns= ["genre"]) #input
print(X)

y = music_data['genre'] #output
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score


#calculating accuracy testing (70-80% data) and training (20-30% data)
