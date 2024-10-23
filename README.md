# MeXE402_Midterm_Almonte-RayIvanC.--Blasco-TreatySherrizahL.

## Introduction

Every day, data is collected and stored. These data can help shape the very world we live in whether it makes the world better, makes life better, or can make life and the world more secure and especially, safer. It can also help with predicting outcomes or results whether it be about behavior or risks and that is where Machine Learning can be applied. Machine Learning is the technological advancement that allows computers to learn from data or variables it is given and execute certain tasks it is being asked to do. One of these tasks is being able to predict a certain outcome based on the data or the variables and what the results were based on those. This kind of task is called Regression Analysis wherein there are factors, called the independent variables, that lead to outcomes, called dependent variables. The dependent variable is what the machine is trying to predict based on the numerous independent variables that are factors to that outcome. For this project, there are two kinds of Regression analysis used which are the Linear Regression Analysis and the Logistic Regression analysis. Linear Regression Analysis refers to the relationship between the independent variables and their impact on the dependent variable which is classified to be in continuous nature. Logistic Regression Analysis is also known as a logit model, is a statistical analysis method to predict a binary outcome, such as yes or no, true or false and 1 or 0 based on a prior observation of a dataset. A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables.

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
+ Estimating the value of a dependent variable for a given set of independent variables.
+ Understanding relationships in determining the strength and nature of the relationships between variables.
+ Developing models by creating mathematical models to represent the relationships between variables.
+ Evaluating models by assessing the goodness of fit and predictive power of the models.


# Linear Regression Analysis

For **Linear Regression Analysis**, the dataset used is the "Life Expectancy (WHO)" which is in a format of **comma separated values**. With the given data (which are now interpreted as variables), the machine is able to assess and predict the Life Expectancy based on inputs that it will be given. The following section was the overall step-by-step coding process.

## Methodology

**Figure 1.1: Importing ‘Pandas’ and creating a dataset for analysis.**
![image](https://github.com/user-attachments/assets/eb0674f4-5950-4817-a14c-b580f8de85f0)

+ import pandas as pd: This line imports the Pandas library, which is a powerful tool for data manipulation and analysis in Python.
+ dataset = pd.read_csv('Life_Expectancy.csv'): This line reads a CSV file named "Life_Expectancy.csv" and stores it in a variable called dataset. The ‘pd.read_csv()’ function from Pandas is used for this.

 
**Figure 1.2: Existing columns within the dataset.**
![image](https://github.com/user-attachments/assets/6b8ef250-8273-49c1-90a1-0fcda426bc61)


**Figure 1.3: Missing (blank) values within each column of the dataset.**
![image](https://github.com/user-attachments/assets/95def765-783d-47a3-862d-c05849a0a79e)

+ missing_data = dataset.isna(): This line creates a new Data Frame (missing_data) that indicates which cells in the original dataset are missing values.
+ missing_count = missing_data.sum(): This line counts the number of missing values in each column of the dataset.
+ missing_count: This line shows the number of missing values in each column.

**Figure 1.4: Fills in missing values (blank cells) within the dataset's existing columns.**
![image](https://github.com/user-attachments/assets/0c106e1c-4329-4566-b393-b648e9cf9500)

dataset[''] = dataset[''].fillna(dataset[''].mean()): 
+ dataset['']: This part selects the column from the dataset.
+ .fillna(): This method is used to fill in missing values.
+ dataset[''].mean(): This calculates the mean (average) value of the column, excluding any missing values.


**Figure 1.5: Displays the first 5 rows of a Pandas Data Frame named dataset.**
![image](https://github.com/user-attachments/assets/f36dd3bf-ef9f-4d32-a4dc-cb343ea16b80)


**Figure 1.6: Imports the NumPy library and sets print options for better formatting of numerical output.**
 ![image](https://github.com/user-attachments/assets/576c569c-a348-4e7c-93a4-0cfa067ec4ae)

+ import numpy as np: This line imports the NumPy library, which is a powerful tool for numerical computations in Python.
+ np.set_printoptions(precision=0, suppress=True, formatter={'all': '{:..ef}'.format()}): This line sets specific printing options for NumPy arrays:
  + precision=0: Sets decimal places to 0 for floating-point numbers. 
  + suppress=True: Suppresses scientific notation for large or small numbers. 
  + formatter={'all': '{:..ef}'.format()}: Specifies a custom ‘formatter’ for all elements to ensure consistent and readable output.
 

**Figure 1.7: Extracts a specific subset of columns from a Pandas Data Frame named dataset and converts the resulting data into a NumPy array.**

![image](https://github.com/user-attachments/assets/932291cb-d3d9-409f-bb94-685271081b57)
 
dataset.iloc[:, 3:-1]: This part selects a specific range of columns from the dataset Data Frame. ‘:’ indicates selecting all rows. ‘3:-1’ specifies the column range starting from the 4th column and excluding the last column

**Figure 1.8: Extracts the last column from a Pandas Data Frame named dataset and converts it into a NumPy array.**

![image](https://github.com/user-attachments/assets/cd0b163d-58b0-4e8f-92b0-26dca0eb175a)

dataset.iloc[:, -1]: This part selects the last column from the dataset Data Frame. ‘iloc[:, -1]’ means selecting all rows and the last column.


**Figure 1.9: Splits a dataset into training and testing sets for machine learning.**

![image](https://github.com/user-attachments/assets/dcb526bf-2a41-479f-8079-50b313d534d3)

+ from sklearn.model_selection import train_test_split: This line imports the ‘train_test_split’ function from the ‘sklearn.model_selection’ module. This function is used to randomly split a dataset into training and testing sets.   
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0): This line splits the dataset into training and testing sets: 
  + X and y: Input data (features) and output data (target variable). 
  + test_size=0.2: 20% of data for testing, 80% for training. 
  + random_state=0: Sets a random seed for reproducible splits.

**Figure 1.10: hows the results of a training set split in Python with variables X and y.**

![image](https://github.com/user-attachments/assets/fdef5d25-63f6-4b6a-b31e-5a23fd0ed579)
 
+ X_train: This represents the training set for the input features (independent variables). 
+ y_train: This represents the training set for the target variable (dependent variable).

**Figure 1.11: Shows the results of a testing set split in Python with variables X and y.**

 ![image](https://github.com/user-attachments/assets/c4da0af8-9e56-430d-b5ce-3f1d48a1171e)

+ X_test: This represents the testing set for the input features (independent variables). 
+ y_test: This represents the testing set for the target variable (dependent variable).

**Figure 1.12: Creates a linear regression model object.**

![image](https://github.com/user-attachments/assets/6763a7e8-dd73-47b3-8ea3-001a1997d5b7)

+ from sklearn.linear_model import LinearRegression: This line imports the ‘LinearRegression’ class from the ‘sklearn.linear_model’ module. This class is used to create linear regression models in Python.
+ model = LinearRegression(): This line creates an instance of the ‘LinearRegression’ class and assigns it to the variable model. This creates a new linear regression model object that you can use for training and making predictions.

**Figure 1.13: Trains a linear regression model on a training dataset.**

![image](https://github.com/user-attachments/assets/739a0ba0-9a74-45a9-9308-22c279b232aa)

model.fit(X_train, y_train): This line trains the linear regression model using the training data: 
+ X_train: This is the training set for the input features (independent variables)
+ y_train: This is the training set for the target variable (dependent variable).

**Figure 1.14: Makes predictions using a trained linear regression model on a testing dataset.**

![image](https://github.com/user-attachments/assets/b914445d-19fd-4438-bd13-1a652a05df09)
 
+ y_pred = model.predict(X_test): This line makes predictions on the testing set using the trained linear regression model. This method uses the model's learned parameters to calculate predicted values for the target variable based on the input features in the testing set
+ y_pred: This variable stores the predicted values generated by the model.

**Figure 1.15: Makes predictions using a trained linear regression model on a specific set of input data.**

 ![image](https://github.com/user-attachments/assets/1a710c33-a8fe-43eb-9f67-90f8d6bb0db8)

+ LE = model.predict([[214, 54, 4.52, 31.27232188, 67, 58848, 24.8, 68, 77, 4.71, 67, 0.1, 2842.938353, 112249, 1, 9.7, 0.676, 11.7]]):
  + LE is the variable that will store the predicted value and ‘model.predict()’ is a function used to make predictions using the trained model along with a list containing a single array of input values. Each value in the array represents a feature or independent variable. The model will use these values to make a prediction.



## Summary of Findings

After conducting the procedures in the Methodology, these findings emerged while developing the model. The following are significant variables for consideration and helped in shaping the model into the way it is and how it resulted.

1. **Incomplete Data**: The datasheet “Life Expectancy (WHO)” contained several missing data. By using the code from Figure 1.3, the following figure shows how many cells have no entry.
**Figure 1.16: Number of missing values in each existing column**

![image](https://github.com/user-attachments/assets/64c8adee-9190-4af5-85bb-cebe7abde6cc)


3. **Prejudiced Variable Influence**: Noticeably in the datasheet, some trends are followed and somewhat consistent through numerous sets of data. The following figures show these trends.

**Set 1.1: Scatter Plot of every variable and the outcome (Life Expectancy)**

**Figure 1.17: Adult Mortality and Life Expectancy**
![image](https://github.com/user-attachments/assets/434b1855-5413-45ef-a21c-fb5fab7355cf)


**Figure 1.18: Infant Deaths and Life Expectancy**
![image](https://github.com/user-attachments/assets/815a389f-264b-4cf4-9214-0e403ca6e5ff)


**Figure 1.19: Alcohol and Life Expectancy**
![image](https://github.com/user-attachments/assets/640b8d7f-9e76-4e24-a626-d02495e2c590)


**Figure 1.20: Percentage Expenditure and Life Expectancy**
![image](https://github.com/user-attachments/assets/c68f6440-2f15-4811-bd3f-7121712cf586)


**Figure 1.21: Hepatitis B Immunity and Life Expectancy**
![image](https://github.com/user-attachments/assets/1426d02b-e6ce-4998-a984-853b9cd7d7b3)


**Figure 1.22: Measles Immunity and Life Expectancy**
![image](https://github.com/user-attachments/assets/7101ec61-daa2-48f1-be2f-51e2aa178081)


**Figure 1.23: BMI and Life Expectancy**
![image](https://github.com/user-attachments/assets/3e251659-b9bb-4e65-84f1-9b1908eb0c2c)


**Figure 1.24: Under-five deaths and Life Expectancy**
![image](https://github.com/user-attachments/assets/d7a33775-ada5-43ef-b94d-4280fad8ab89)


**Figure 1.25: Polio Immunity and Life Expectancy**
![image](https://github.com/user-attachments/assets/3b31896d-45b9-4885-9699-a2e6567e5d3e)


**Figure 1.26: Total Expenditure and Life Expectancy**
![image](https://github.com/user-attachments/assets/017ea577-064d-45fb-9f24-625f5f0e1732)


**Figure 1.27: Diphtheria Immunity and Life Expectancy**
![image](https://github.com/user-attachments/assets/3024938d-7b65-45c9-820c-afb85c4f889c)


**Figure 1.28: HIV/AIDS and Life Expectancy**
![image](https://github.com/user-attachments/assets/0f438030-c709-4753-963b-466ac82940cc)


**Figure 1.29: GDP and Life Expectancy**
![image](https://github.com/user-attachments/assets/843442f4-f666-41c4-a8cc-82b39467354c)


**Figure 1.30: Population and Life Expectancy**
![image](https://github.com/user-attachments/assets/bcc98c75-65a2-4b9b-8a24-eafa4783e6bd)


**Figure 1.31: Thinness among 10-19 years and Life Expectancy**
![image](https://github.com/user-attachments/assets/e0e05107-869b-485f-b00e-e3d5325209d9)


**Figure 1.32: Thinness among 5-9 years and Life Expectancy**
![image](https://github.com/user-attachments/assets/5b372c1e-f9f3-4e31-8873-1e8d9a46f60d)


**Figure 1.33: Income composition of resources and Life Expectancy**
![image](https://github.com/user-attachments/assets/3ceaa178-68da-44ed-8b5c-2c8e36c618e2)


**Figure 1.34: Schooling and Life Expectancy**
![image](https://github.com/user-attachments/assets/d7a647bf-4299-4be9-a115-dac0d2de1574)


3. **Model Evaluation**: After the model has been successfully made and is functioning based on the code shown in Figure 1.15, a test for evaluation is due for execution, and as such, R-squared, Adjusted R-Squared, and Mean Squared Error were used.
+ **R-squared**
	+ R-squared measures the proportion of variance in the dependent variable that is explained by the independent variables in the model. Values range from 0 to 1. A higher R-squared indicates a better fit of the model to the data, meaning the model explains more variability in the outcome.
+ **Adjusted R-squared**
	+ Adjusted R-squared adjusts the R-squared value based on the number of predictors in the model, penalizing for adding non-informative variables. It provides a more accurate measure of model fit, especially when comparing models with different numbers of predictors. A higher Adjusted R-squared indicates a better model that explains variability without being overly complex.
+ **Mean Squared Error**
	+  MSE calculates the average of the squared differences between actual and predicted values, quantifying the model's prediction error. Lower MSE values indicate better predictive accuracy. It reflects how close the model's predictions are to the actual outcomes, with smaller values representing less error.

**Figure 1.35: Calculates the R-squared score**
![image](https://github.com/user-attachments/assets/b6e5a3de-d029-41ff-8f92-66ee51a2cb30)


**Figure 1.36: Calculates the adjusted R-squared score**
![image](https://github.com/user-attachments/assets/e00c693a-a1c9-475d-a7b8-7bcbbcb262a3)


**Figure 1.37: Calculates the Mean Squared Error**
![image](https://github.com/user-attachments/assets/0ee6fd3d-cdf5-4134-8d68-817b1d86fbe2)


## Logistic Regression Analysis
For Logistic Regression Analysis, the dataset used is the "Telco Churn Prediction" which is in a format of **comma-separated values.** With the given data (which are now interpreted as variables), the machine is able to assess and predict the probability of a customer churning (leaving the service) based on the inputs provided. The following section was the overall step-by-step coding process.

## Methodology

**Figure 2.1: The code snippet loads the CSV file "telcoduplic.csv" into a pandas DataFrame called dataset and displays the first 5 or 10 rows for a quick overview.**
![image](https://github.com/user-attachments/assets/2659b4c2-d72b-4d48-97ba-75649090a728)

+ Importing the pandas library: pandas is a powerful Python library for data manipulation and analysis. It's essential for working with datasets.
+ Reading the CSV file: The read_csv() function from pandas is used to read the CSV file named 'telcoduplic.csv' and store its contents in a DataFrame called dataset.
+ Displaying the first few rows: The head() method is used to print the first 5 rows of the DataFrame, providing a quick overview of the data's structure and content.

**Figure 2.2: The code snippet extracts the input features (X) and the target variable (y) from the dataset.**
![image](https://github.com/user-attachments/assets/04842060-498c-4cab-8d71-8099017264f2)

+ Importing NumPy: numpy is a fundamental Python library for numerical operations. It's used for efficient array manipulation.
+ Setting print options: This code adjusts how numbers are displayed to improve readability.
+ Extracting input features (X): 
	+ dataset.iloc[:, 1:-1] selects all rows and columns from the second column (index 1) to the second-to-last column (excluding the last column). This creates a new array containing the input features.
	+ .values converts the DataFrame to a NumPy array for numerical operations.
+ Extracting target variable (y): 
	+ dataset.iloc[:, -1] selects all rows and the last column, containing the target variable.
	+ .values converts it to a NumPy array.

**Figure 2.3: The code snippet splits the dataset into training and testing sets, with 20% for testing and a fixed random seed for reproducibility.**
![image](https://github.com/user-attachments/assets/944aa7ad-4446-4dc4-adcd-eb4754a1e417)

+ Importing train_test_split: This function from the sklearn.model_selection module is used to divide the data into training and testing sets.
+ Splitting the data: train_test_split(X, y, test_size=0.2, random_state=0) splits the input features (X) and the target variable (y) into training and testing sets. 
  + test_size=0.2 indicates that 20% of the data will be used for testing.
  + random_state=0 ensures reproducibility by setting a fixed random seed for the splitting process.
+ Displaying the training and testing sets: The code prints the first few rows of the X_train, X_test, y_train, and y_test arrays to visualize the split data.

**Figure 2.4: The code snippet scales the training data features to have zero mean and unit variance.**
![image](https://github.com/user-attachments/assets/09197be7-acb6-4f2c-9ca1-22cd1b7ffd71)

+ Importing StandardScaler: This class from the sklearn.preprocessing module is used for standardizing features by subtracting the mean and dividing by the standard deviation.
+ Creating a StandardScaler object: sc = StandardScaler() creates an instance of the StandardScaler class.
+ Fitting and transforming the training data: X_train = sc.fit_transform(X_train) standardizes the features in the training set X_train. The fit_transform() method calculates the mean and standard deviation of the training data and applies the scaling transformation.

**Figure 2.5: The code snippet creates a logistic regression model with a random state of 0 for reproducibility.**
![image](https://github.com/user-attachments/assets/16b914a1-a6ac-4f41-aaec-ec04a2ab10df)

+ Importing LogisticRegression: This class from the sklearn.linear_model module is used to create a logistic regression model.
+ Creating a logistic regression model: model = LogisticRegression(random_state=0) creates an instance of the LogisticRegression class with a random state of 0. The random state ensures reproducibility by setting a fixed random seed for the model's initialization.

**Figure 2.6: The code snippet trains the logistic regression model on the training data.**
![image](https://github.com/user-attachments/assets/7f931248-8ff8-405d-a428-d82f8d76b73c)

+ Training the model: model.fit(X_train, y_train) trains the logistic regression model model on the training set X_train and y_train. The model learns the relationship between the input features and the target variable.

**Figure 2.7: The code snippet makes predictions on the testing set using the trained logistic regression model.**
![image](https://github.com/user-attachments/assets/e91ac424-38ca-461e-84fe-5ffcf3bcd983)

+ Making predictions: y_pred = model.predict(sc.transform(X_test)) uses the trained model model to predict the target variable for the testing set X_test. Before making predictions, the testing data is transformed using the same StandardScaler object sc that was used for the training data.
+ Displaying predictions: The code prints the predicted values y_pred, which are the probabilities of belonging to the positive class for each data point in the testing set.
+ Making a single prediction: model.predict(sc.transform([[0,0,0,0,21,1,0,0,0,0,0,0,0,0,0,1,2,68.65,1493.2]])) makes a prediction on a single new data point. The input data must be transformed using the same StandardScaler before being passed to the model.

**Figure 2.8: The code snippet evaluates the logistic regression model's performance using a confusion matrix and accuracy score.**
![image](https://github.com/user-attachments/assets/248fc56e-dcb3-49b4-8efc-53e80be95cea)

+ Confusion Matrix: 
   + confusion_matrix(y_test, y_pred) calculates the confusion matrix, which shows the correct and incorrect predictions for each class.
   + The resulting array shows the number of true positives, false positives, false negatives, and true negatives.

+ Accuracy: 
   + accuracy_score(y_test, y_pred) calculates the accuracy of the model, which is the proportion of correct predictions.

## Summary of Findings
**Figure 2.9: The Evaluation Metrics: Calculating the accuracy**
![image](https://github.com/user-attachments/assets/bd791a6b-1680-473c-a379-b3b0a78311f3)

**Figure 2.10: The visualization of plotting the confusion matrices**

![image](https://github.com/user-attachments/assets/a2bb7fc0-db58-464f-861a-3820861f6582)

**Figure 2.11: Box Plot of Churn Vs Tenure**

![image](https://github.com/user-attachments/assets/7bafdc8a-2a87-4315-bb82-3bff1cceeb8c)

**Figure 2.12: Density Plot of Churn by Monthly Charges**

![image](https://github.com/user-attachments/assets/1cadf791-525a-4191-ad18-7b7062d1e077)

**Figure 2.13: Density Plot of Churn by Total Charges**

![image](https://github.com/user-attachments/assets/df0ebb83-a849-4006-8ea1-a44bb138939a)


## Discussion

Discussion of the continuous variables tenure, monthly charges, and total charges to see how they vary by other variables.

**Churn vs Tenure:**  As we can see form the churn vs tenure plot, the customers who do not churn, they tend to stay for a longer tenure with the telecom company. 

**Churn by Monthly Charges:**  Higher % of customers churn when the monthly charges are high.

**Churn by Total Charges:** It seems that there is higher churn when the total charges are lower.

### Interpreting the Values in Confusion Matrix
+ **True Positives (TP):** 950 - The model correctly predicted 950 instances of class 0.
+ **False Negatives (FN):** 164 - The model incorrectly predicted 164 instances of class 0 as class 1.
+ **False Positives (FP):** 99 - The model incorrectly predicted 99 instances of class 1 as class 0.
+ **True Negatives (TN):** 196 - The model correctly predicted 196 instances of class 1.

**Figure 2.14: Calculating the matrix**

![image](https://github.com/user-attachments/assets/484edb80-5640-4968-a61b-5eadc2126f7d)

+ **Accuracy:** 81.3% of the predictions were correct overall.
+ **Precision:** 90.5% of the instances predicted as class 0 were actually class 0.
+ **Recall:** 85.3% of the actual class 0 instances were correctly predicted.
+ **F1-score:** The model achieved a balanced performance, with an F1-score of 87.8%.

	These metrics suggest that the model is performing reasonably well, but there is still room for improvement, especially in terms of recall. It's important to consider the specific context of the problem and the relative importance of precision and recall when evaluating the model's performance.


































