import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)

# Saving the model
import pickle
pickle.dump(model, open('boston_houses.pkl', 'wb'))