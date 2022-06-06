from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

# from imblearn.over_sampling import SMOTE # pip install imbalanced-learn
import datetime


# Read input
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Data cleansing
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

labels = train["Survived"].values.astype('int32')

test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
test.at[152, "Fare"] = test.Fare.median()

# pre-processing: divide by max and substract mean
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]

for feature in features:

    scale = np.max(train[feature])
    train[feature] /= scale
    test[feature] /= scale

    mean = np.std(train[feature])
    train[feature] -= mean
    test[feature] -= mean

X_train = train[features].values.astype('float32')
X_test = test[features].values.astype('float32')

# smote = SMOTE(random_state=42)
# X_train, labels = smote.fit_resample(X_train, labels)

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels)

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]
epochs = 100

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

print("Training...")
model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1, verbose=1)

preds = np.argmax(model.predict(X_test), axis=-1)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
solution = pd.DataFrame(preds, PassengerId, columns=["Survived"])

# my_tree_one.csvとして書き出し
dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
solution.to_csv("./titanic_{}.csv".format(dt_now), index_label=["PassengerId"])
