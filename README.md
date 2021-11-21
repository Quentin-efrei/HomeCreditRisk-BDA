# HomeCreditRisk-BDA
Big Data Applications - HomeCreditRiskAssessment

Project's autor : Hélène Boersma, Quentin Courtois & Guillaume Jaouen

## Introduction

The objective of this project was to make predictions from the Home Credit Risk Classification dataset. The dataset is made up of several csv files. For this project, we focused on application_train.csv on which we built our models, then we made our predictions on application_test.csv. The application_train.csv file is composed of 122 columns, the TARGET column represents whether an applicant is able to repay a loan. The application_test.csv file contains the same columns except TARGET which must be predicted. <br>
Here we made our predictions following 3 different models: Xgboost, Random Forest and Gradient Boosting

## 1. Machine Learning part

### 1.1  Data preparation

For the Data preprocessing part, there were a lot of columns with missing values. We therefore decided to remove the columns containing more than 30% of missing values thanks to the function missing_values (). <br>
We also chose to remove all the ‘AMT_REQ_CREDIT_BUREAU_XXX’ columns because it did not seem necessary to make our predictions. <br>
For the columns that contained less than 30% missing values, we replaced the missing values with the means for the numeric values and with a new categorical value for the qualitative values. <br>
In addition, we made dummies on categorical values. In fact, one-hot encoding allows us to assign binary values to categorical data. Thanks to this, better predictions can be made because one value is not given more weight than another. <br>
Finally, we normalize all data with sklearn's MinMaxScaler function. This is to put all the data on the same scale and therefore equalize the weights of each dimension. <br> 


### 1.2 Feature engineering


### 1.3 Models training 


### 1.4 Predictions


### 1.5 Sphinx library


## 2. MLFlow library part

- MLFlow

## 3. SHAP interpretation part

- SHAP interpretation
