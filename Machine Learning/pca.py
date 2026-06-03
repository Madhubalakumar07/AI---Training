import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Load dataset

data = pd.read_csv(r"D:\Navigate lab\Datasets\heart.csv")
print(data.head())

# Prepare data
X = data.drop('HeartDisease', axis='columns')
y = data['HeartDisease']

# Encode categorical variables
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])
print(X.head())

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

# Check before PCA
print(x_train.shape, x_test.shape)
model_grid = {
    'Logistic_reg':( LogisticRegression(), {'max_iter': [1000]}),
    'svc': (SVC(), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}),
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [None, 10]})
}
best_score_before_pca = []
for model_name, (model, params) in model_grid.items():
    print(f'Tuning {model_name}...')
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    score = best_model.score(x_test, y_test)
    best_score_before_pca.append(score)
    print(f'{model_name} Best Parameters: {grid_search.best_params_}')
print(f'Best score before PCA: {best_score_before_pca}\n')

# Apply PCA
pca = PCA(0.95)  # Keep 95% of variance
x_pca = pca.fit_transform(x_scaled)
x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
print(x_train_pca.shape, x_test_pca.shape)

# Check after PCA
best_score_after_pca = []
for model_name, (model, params) in model_grid.items():
    print(f'Tuning {model_name} after PCA...')
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(x_train_pca, y_train)
    best_model = grid_search.best_estimator_
    score = best_model.score(x_test_pca, y_test)
    best_score_after_pca.append(score)
    print(f'{model_name} Best Parameters after PCA: {grid_search.best_params_}')
print(f'Best score after PCA: {best_score_after_pca}\n')