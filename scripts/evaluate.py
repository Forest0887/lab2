import json
import math
import pickle
import sys

import pandas as pd
from sklearn.metrics import mean_squared_error

from storage import Storage

x_test = pd.read_csv(str(sys.argv[1]))
y_test = pd.read_csv(str(sys.argv[2]))

logistic_model = pickle.load(open(str(Storage.MODELS_PATH / "logistic.pickle"), "rb"))

logistic_r_squared = logistic_model.score(x_test, y_test)

logistic_y_pred = logistic_model.predict(x_test)
logistic_rmse = math.sqrt(mean_squared_error(y_test, logistic_y_pred))

forest_model = pickle.load(open(str(Storage.MODELS_PATH / "forest.pickle"), "rb"))

forest_r_squared = forest_model.score(x_test, y_test)

forest_y_pred = forest_model.predict(x_test)
forest_rmse = math.sqrt(mean_squared_error(y_test, forest_y_pred))

sgdr_model = pickle.load(open(str(Storage.MODELS_PATH / "sgdr.pickle"), "rb"))

sgdr_r_squared = sgdr_model.score(x_test, y_test)

sgdr_y_pred = sgdr_model.predict(x_test)
sgdr_rmse = math.sqrt(mean_squared_error(y_test, sgdr_y_pred))

with open(str(Storage.RESULT_FILE_PATH), "w") as result:
    json.dump(
        dict(
            logistic=dict(r_squared=logistic_r_squared, rmse=logistic_rmse), 
            forest=dict(r_squared=forest_r_squared, rmse=forest_rmse), 
            sgdr=dict(r_squared=sgdr_r_squared, rmse=sgdr_rmse)
        ), 
    result)
