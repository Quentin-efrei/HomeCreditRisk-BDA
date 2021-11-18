import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

def missing_values(df):
    '''
    Functions that return columns with more than 30% of missing values
    '''
    total_missing = df.isnull().sum()/df.shape[0] #total missing value of df
    percent_missing = total_missing*100 # putting total_missing as a percentage
    return percent_missing.sort_values(ascending=False).round(1) # return every column that are missing 30% sorted by descending.

def drop_some_columns(df):
    '''
    function that drop following columns that after ur analysis dont seem relevant.
    '''
    data = df.drop(columns=['AMT_REQ_CREDIT_BUREAU_YEAR',
                            'AMT_REQ_CREDIT_BUREAU_HOUR',
                            'AMT_REQ_CREDIT_BUREAU_DAY',
                            'AMT_REQ_CREDIT_BUREAU_WEEK',
                            'AMT_REQ_CREDIT_BUREAU_MON',
                            'AMT_REQ_CREDIT_BUREAU_QRT']) #dropping those columns in the dataframe df
    return data # return the dataframe.

def fill_some_rows(df):
    '''
    Function to fill rows that are missing value, categorical or numerical.
    '''
    data = df #copying dataframe
    data['AMT_GOODS_PRICE'] = data['AMT_GOODS_PRICE'].fillna(data['AMT_GOODS_PRICE'].mean()) # filling with means
    data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].fillna("Unaccompanied") # filling with Unacompannied
    data['EXT_SOURCE_2'] = data['EXT_SOURCE_2'].fillna(data['EXT_SOURCE_2'].mean()) # filling with mean
    data['EXT_SOURCE_3'] = data['EXT_SOURCE_3'].fillna(data['EXT_SOURCE_3'].mean()) # filling with mean
    data['OBS_30_CNT_SOCIAL_CIRCLE'] = data['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(data['OBS_30_CNT_SOCIAL_CIRCLE'].mean()) # filling with mean
    data['OBS_60_CNT_SOCIAL_CIRCLE'] = data['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(data['OBS_60_CNT_SOCIAL_CIRCLE'].mean()) # filling with mean
    data['DEF_30_CNT_SOCIAL_CIRCLE'] = data['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(data['DEF_30_CNT_SOCIAL_CIRCLE'].mean()) # filling with mean
    data['DEF_60_CNT_SOCIAL_CIRCLE'] = data['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(data['DEF_60_CNT_SOCIAL_CIRCLE'].mean()) # filling with mean
    data['AMT_ANNUITY'] = data['AMT_ANNUITY'].fillna(data['AMT_ANNUITY'].mean()) # filling with mean
    data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'].fillna(data['CNT_FAM_MEMBERS'].mean()) # filling with mean
    data['DAYS_LAST_PHONE_CHANGE'] = data['DAYS_LAST_PHONE_CHANGE'].fillna(data['DAYS_LAST_PHONE_CHANGE'].mean()) # filling with mean
    return data # return updated dataframe


def make_categorical_numerical(df):
    '''
    Function to make dummies & put categorical as numerical values
    '''
    data = df
    data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].astype('category').cat.codes # category to numerical in same column
    data['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE'].astype('category').cat.codes # category to numerical in same column
    data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].astype('category').cat.codes # category to numerical in same column
    data['WEEKDAY_APPR_PROCESS_START'] = data['WEEKDAY_APPR_PROCESS_START'].astype('category').cat.codes # category to numerical in same column
    data['ORGANIZATION_TYPE'] = data['ORGANIZATION_TYPE'].astype('category').cat.codes # category to numerical in same column
    data['CODE_GENDER'] = data['CODE_GENDER'].astype('category').cat.codes # category to numerical in same column
    data['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE'].astype('category').cat.codes # category to numerical in same column
    data['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS'].astype('category').cat.codes # category to numerical in same column
    data = pd.get_dummies(df) # transforming rest of categorical values as dummies
    return data # returning updated dataframe.

def scale_data(df):
    '''
    Function to normalize entire dataset
    '''
    min_max_scaler = MinMaxScaler() # using MinMaxScaler of sklearn
    data = df  # copying dataset
    for col in data: # looping through the dataset
        data[[col]] = min_max_scaler.fit_transform(data[[col]]) # applying normalization to each column.
    return data # returning updated dataframe

def train_random_forest(X_train,y_train):
    '''
    Function to train a random forest model
    '''
    model = RandomForestClassifier(n_estimators=100) #initializing randomforest classifier with n_estimators = 100
    model.fit(X_train,y_train) # fitting data to the model.
    return model # returning created model

def train_xgboost(X_train,y_train):
    '''
    Function to train a xgboost model
    '''

def train_gradient_boosting(X_train,y_train):
    '''
    Function to train a gradient boosting model
    '''

def predictions(model,X_test):
    '''
    Function that return prediction of a model based on a test dataset
    '''
    predictions = model.predict(X_test) # doing predictions on X_test
    return predictions # returning array of predictions

def accuracy(predictions,y_test):
    '''
    Function that print performance of a model based on validation dataset
    '''
    print('Accuracy of model : ',metrics.accuracy_score(predictions,y_test)) #printing accuracy of model

