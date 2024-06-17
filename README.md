# titanic-ML
Titanic Survival Prediction with Logistic Regression
This project involves predicting the survival of passengers aboard the Titanic using logistic regression, a supervised learning technique.

Dataset
The dataset (train.csv) used in this project contains information about passengers, such as age, sex, class, and whether they survived the Titanic disaster.

Workflow
Data Loading and Exploration:

The dataset is loaded into a Pandas DataFrame (titanic_data), providing insights into the structure and initial rows.
Basic exploratory data analysis (EDA) includes checking dimensions, data types, and detecting missing values.
Data Cleaning and Preprocessing:

The "Cabin" column, which has many missing values, is dropped from the dataset.
Missing values in the "Age" column are imputed with the mean age of passengers.
Missing values in the "Embarked" column are filled with the mode (most frequent value).
Exploratory Data Analysis (EDA):

Statistical summaries (describe() function) and visualizations (using sns.countplot) provide insights into survival rates, gender distribution, class distribution, and more.
Categorical columns "Sex" and "Embarked" are converted to numerical values for model compatibility.
Model Training and Evaluation:

The dataset is split into training and test sets using train_test_split.
Logistic regression is employed to build a predictive model for survival prediction.
Training accuracy and test accuracy are computed to evaluate the model's performance.
Prediction:

The trained logistic regression model is used to predict survival outcomes for new data instances (passenger profiles).
Example predictions are made on both training and test datasets, with accuracy scores reported.
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn (sklearn)
Usage
Ensure Python and necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn) are installed.
Clone this repository and navigate to the project directory.
Place your dataset (train.csv) in the project directory.
Run the script titanic_survival_prediction.py to load the dataset, preprocess data, train the logistic regression model, and predict survival outcomes based on new passenger profiles.
Example
Upon running the script, the logistic regression model leverages passenger data (such as age, sex, and class) to predict survival outcomes for passengers aboard the Titanic. The accuracy of predictions on both training and test datasets provides an assessment of the model's reliability in predicting survival.

