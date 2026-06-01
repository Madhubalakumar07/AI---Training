import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
# Load dataset
digits = load_digits()
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target
print(df.head())

# Split dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_score = knn.score(x_test, y_test)
print(f'KNN Accuracy: {knn_score:.2f}')

#confusion matrix
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))