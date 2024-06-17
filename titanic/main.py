import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the data from CSV file into a Pandas DataFrame
titanic_data = pd.read_csv('train.csv')

# Displaying the first few rows of the dataframe
titanic_data.head()

# Checking the number of rows and columns in the dataframe
titanic_data.shape

# Getting information about the dataset
titanic_data.info()

# Checking for missing values in each column
titanic_data.isnull().sum()

# Dropping the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# Replacing missing values in "Age" column with the mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Finding the mode value of "Embarked" column
print("Mode of Embarked column:", titanic_data['Embarked'].mode())
mode_embarked = titanic_data['Embarked'].mode()[0]

# Replacing missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(mode_embarked, inplace=True)

# Checking again for missing values in each column
titanic_data.isnull().sum()

# Displaying statistical measures about the data
titanic_data.describe()

# Finding the number of people who survived and did not survive
titanic_data['Survived'].value_counts()

# Setting seaborn style
sns.set()

# Creating a count plot for "Survived" column
sns.countplot('Survived', data=titanic_data)

# Counting values in "Sex" column
titanic_data['Sex'].value_counts()

# Creating a count plot for "Sex" column
sns.countplot('Sex', data=titanic_data)

# Number of survivors gender-wise
sns.countplot('Sex', hue='Survived', data=titanic_data)

# Creating a count plot for "Pclass" column
sns.countplot('Pclass', data=titanic_data)

# Creating a count plot for "Pclass" column with "Survived" hue
sns.countplot('Pclass', hue='Survived', data=titanic_data)

# Converting categorical columns to numerical values
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Displaying the modified dataframe
titanic_data.head()

# Separating the features and target labels
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Displaying shapes of training and test sets
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Initializing and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predicting on training data and calculating accuracy
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print('Accuracy score on training data:', training_accuracy)

# Predicting on test data and calculating accuracy
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print('Accuracy score on test data:', test_accuracy)
