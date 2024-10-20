# MeXE402_Midterm_Almonte-RayIvanC.--Blasco-TreatySherrizahL.

## Introduction

Every day, data is collected and stored. These data can help shape the very world we live in whether it makes the world better, makes life better, or can make life and the world more secure and especially, safer. It can also help with predicting outcomes or results whether it be about behavior or risks and that is where Machine Learning can be applied. Machine Learning is the technological advancement that allows computers to learn from data or variables it is given and execute certain tasks it is being asked to do. One of these tasks is being able to predict a certain outcome based on the data or the variables and what the results were based on those. This kind of task is called Regression Analysis wherein there are factors, called the independent variables, that lead to outcomes, called dependent variables. The dependent variable is what the machine is trying to predict based on the numerous independent variables that are factors to that outcome. For this project, there are two kinds of Regression analysis used which are the Linear Regression Analysis and the Logistic Regression analysis. Linear Regression Analysis refers to the relationship between the independent variables and their impact on the dependent variable which is classified to be in continuous nature. 

## Dataset Description

The dataset used for the Linear Regression Analysis is the “Life Expectancy (WHO)” which is a Statistical Analysis of factors influencing Life Expectancy.  
The following variables are contained in the datasheet:
+ ***Country*** - 193 unique entries
+ ***Year*** - Ranging from 2010 to 2015
+ ***Status*** - Developed or Developing status
+ ***Population***
+ ***Life expectancy*** - Life Expectancy in age
+ ***Adult Mortality*** - Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
+ ***Infant deaths*** - Number of Infant Deaths per 1000 population
+ ***Alcohol*** - Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
+ ***Percentage expenditure*** - Expenditure on health as a percentage of Gross Domestic Product per capita(%)
+ ***Hepatitis B*** - Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
+ ***Measles*** - Measles - number of reported cases per 1000 population
+ ***BMI*** - Average Body Mass Index of entire population
+ ***Under-five deaths*** - Number of under-five deaths per 1000 population
+ ***Polio*** - Polio (Pol3) immunization coverage among 1-year-olds (%)
+ ***Total expenditure*** - General government expenditure on health as a percentage of total government expenditure (%)
+ ***Diphtheria*** - Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
+ ***HIV/AIDS*** - Deaths per 1 000 live births HIV/AIDS (0-4 years)
+ ***GDP*** - Gross Domestic Product per capita (in USD)
+ ***Thinness 10-19 years*** - Prevalence of thinness among children and adolescents for Age 10 to 19 (%)
+ ***Thinness 5-9 years*** - Prevalence of thinness among children for Age 5 to 9 (%)
+ ***Income composition of resources*** - Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
+ ***Schooling*** - Number of years of Schooling (years)













The dataset used for Logistic Regression Analysis contains customer information from a telecommunication company, including whether they have churned of not. 
The data set includes information about:
+ Customers who left within the last month – the column is called Churn
+ Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
+ Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
+ Demographic info about customers – gender, age range, and if they have partners and dependents




## Project Objectives

# Regression Analysis

## Linear Regression Analysis

**This code cell is for importing ‘Pandas’ and creating a dataset for analysis.**


+ import pandas as pd: This line imports the Pandas library, which is a powerful tool for data manipulation and analysis in Python.
+ dataset = pd.read_csv('Life_Expectancy.csv'): This line reads a CSV file named "Life_Expectancy.csv" and stores it in a variable called dataset. The ‘pd.read_csv()’ function from Pandas is used for this.

 
dataset.columns: This code is simply the existing columns within the dataset.

**This code is used to identify and count missing (blank) values within each column of a Pandas Data Frame named dataset.**
 
+ missing_data = dataset.isna(): This line creates a new Data Frame (missing_data) that indicates which cells in the original dataset are missing values.
+ missing_count = missing_data.sum(): This line counts the number of missing values in each column of the dataset.
+ missing_count: This line shows the number of missing values in each column.

**This code fills in missing values (blank cells) within the 'Adult Mortality' column of a Pandas Data Frame named dataset.**
 
dataset['Adult Mortality'] = dataset['Adult Mortality'].fillna(dataset['Adult Mortality'].mean()): 
•	dataset['Adult Mortality']: This part selects the 'Adult Mortality' column from the dataset Data Frame.
•	.fillna(): This method is used to fill in missing values.
•	dataset['Adult Mortality'].mean(): This calculates the mean (average) value of the 'Adult Mortality' column, excluding any missing values.


**The code dataset.head(5) is used to display the first 5 rows of a Pandas Data Frame named dataset.**
 
dataset.head(5): This refers to the dataset containing the data that shows the first 5 rows of the dataset.

**This code imports the NumPy library and sets print options for better formatting of numerical output.**
 
+ import numpy as np: This line imports the NumPy library, which is a powerful tool for numerical computations in Python.
+ np.set_printoptions(precision=0, suppress=True, formatter={'all': '{:..ef}'.format()}): This line sets specific printing options for NumPy arrays:
  + precision=0: Sets decimal places to 0 for floating-point numbers. 
  + suppress=True: Suppresses scientific notation for large or small numbers. 
  + formatter={'all': '{:..ef}'.format()}: Specifies a custom ‘formatter’ for all elements to ensure consistent and readable output.
 

**This code extracts a specific subset of columns from a Pandas Data Frame named dataset and converts the resulting data into a NumPy array.**
 
dataset.iloc[:, 3:-1]: This part selects a specific range of columns from the dataset Data Frame. ‘:’ indicates selecting all rows. ‘3:-1’ specifies the column range starting from the 4th column and excluding the last column

**This code extracts the last column from a Pandas Data Frame named dataset and converts it into a NumPy array.**
 
dataset.iloc[:, -1]: This part selects the last column from the dataset Data Frame. ‘iloc[:, -1]’ means selecting all rows and the last column.


**This code splits a dataset into training and testing sets for machine learning.**
 
+ from sklearn.model_selection import train_test_split: This line imports the ‘train_test_split’ function from the ‘sklearn.model_selection’ module. This function is used to randomly split a dataset into training and testing sets.   
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0): This line splits the dataset into training and testing sets: 
  + X and y: Input data (features) and output data (target variable). 
  + test_size=0.2: 20% of data for testing, 80% for training. 
  + random_state=0: Sets a random seed for reproducible splits.

**This shows the results of a training set split in Python with variables X and y.**
 
+ X_train: This represents the training set for the input features (independent variables). 
+ y_train: This represents the training set for the target variable (dependent variable).

**This shows the results of a testing set split in Python with variables X and y.**
 
+ X_test: This represents the testing set for the input features (independent variables). 
+ y_test: This represents the testing set for the target variable (dependent variable).

**This code creates a linear regression model object.**
 
+ from sklearn.linear_model import LinearRegression: This line imports the ‘LinearRegression’ class from the ‘sklearn.linear_model’ module. This class is used to create linear regression models in Python.
+ model = LinearRegression(): This line creates an instance of the ‘LinearRegression’ class and assigns it to the variable model. This creates a new linear regression model object that you can use for training and making predictions.

 
**This code trains a linear regression model on a training dataset.**
 
model.fit(X_train, y_train): This line trains the linear regression model using the training data: 
+ X_train: This is the training set for the input features (independent variables)
+ y_train: This is the training set for the target variable (dependent variable).

**This code makes predictions using a trained linear regression model on a testing dataset.**
 
+ y_pred = model.predict(X_test): This line makes predictions on the testing set using the trained linear regression model. This method uses the model's learned parameters to calculate predicted values for the target variable based on the input features in the testing set
+ y_pred: This variable stores the predicted values generated by the model.

**This code makes predictions using a trained linear regression model on a specific set of input data.**
 

+ LE = model.predict([[214, 54, 4.52, 31.27232188, 67, 58848, 24.8, 68, 77, 4.71, 67, 0.1, 2842.938353, 112249, 1, 9.7, 0.676, 11.7]]):
  + LE is the variable that will store the predicted value and ‘model.predict()’ is a function used to make predictions using the trained model along with a list containing a single array of input values. Each value in the array represents a feature or independent variable. The model will use these values to make a prediction.


**This code calculates the R-squared score as a performance metric for a machine-learning model.**
 
+  from sklearn.metrics import r2_score:  Imports the r2_score function for calculating the R-squared score. R-squared score is a statistical measure that indicates how well a regression model fits the data ranging from 0 - 1.
+ r2 = r2_score(y_test, y_pred): This line calculates the R-squared score: 
  + r2_score: This is the function used to calculate the R-squared score.
  + y_test: This is the actual target values from the testing set.
  + y_pred: This is the predicted target values from the model.

**The code calculates the adjusted R-squared score, which is a more accurate measure of model performance when dealing with multiple features.**
 
+ adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1): This line calculates the adjusted R-squared score using the formula: 
  + r2: The R-squared score.
  + n: The number of data points in the testing set.
  + k: The number of features in the model.


## Logistic Regression Analysis
