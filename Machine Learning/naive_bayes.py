import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.datasets import load_wine

# Load dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

# Split dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_score = gnb.score(X_test, y_test)
print(f'Gaussian Naive Bayes Accuracy: {gnb_score:.2f}')

# Train a Multinomial Naive Bayes model
# Note: MultinomialNB is typically used for text data, so we will not use
# it for this dataset, but here is how you would do it:
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb_score = mnb.score(X_test, y_test)
print(f'Multinomial Naive Bayes Accuracy: {mnb_score:.2f}')