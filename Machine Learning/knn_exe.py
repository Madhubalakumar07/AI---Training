import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv(r"D:\Navigate lab\Datasets\diabetes.csv")
print(data.head())

# Prepare data
X = data.drop('Outcome', axis = 'columns')
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train KNN model
k_values = np.arange(1, int(np.sqrt(len(y_train))) + 1)
param_grid = {'n_neighbors': k_values}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best K:", grid_search.best_params_['n_neighbors'])
best_knn = grid_search.best_estimator_
print("test set accuracy:", best_knn.score(x_test, y_test))
print("train set accuracy:", best_knn.score(x_train, y_train))