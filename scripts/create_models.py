import pickle
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression

from storage import Storage

Storage.MODELS_PATH.mkdir(parents=True, exist_ok=True)

x_train = pd.read_csv(str(sys.argv[1]))
y_train = pd.read_csv(str(sys.argv[2]))

# LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
clf = GridSearchCV(LogisticRegression(solver='liblinear'), grid, cv=3, scoring="accuracy")
best = clf.fit(x_train, y_train.to_numpy().ravel()).best_estimator_
best.fit(x_train, y_train.to_numpy().ravel())

pickle.dump(best, open(str(Storage.MODELS_PATH / "logistic.pickle"), "wb"))

# RandomForest
rfc = RandomForestClassifier()
forest = rfc.fit(x_train, y_train.to_numpy().ravel())

pickle.dump(forest, open(str(Storage.MODELS_PATH / "forest.pickle"), "wb"))

# SGDRegressor
sgdr = SGDRegressor(alpha = 0.0001, epsilon = 0.01, eta0 = 0.1, penalty = 'elasticnet')
sgdr = sgdr.fit(x_train, y_train.to_numpy().ravel())

pickle.dump(sgdr, open(str(Storage.MODELS_PATH / "sgdr.pickle"), "wb"))
