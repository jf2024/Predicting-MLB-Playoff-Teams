import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('baseball.csv')

# Histogram of runs scored 
plt.hist(df['RS'])
plt.title('Distribution of Runs Scored')
plt.xlabel('Runs Scored')
plt.ylabel('Count') 
plt.show()

# Scatter plot runs scored vs runs against
plt.scatter(df['RS'], df['RA'])
plt.title('Runs Scored vs Runs Against')
plt.xlabel('Runs Scored')
plt.ylabel('Runs Against')
plt.show()

# Features and Targets
X = df[['RS','RA','OBP','SLG']] 
y = df['Playoffs']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))