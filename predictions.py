import sys
import pickle
import pandas as pd
from functions import scale_data,predictions

def main(argv):
    print('data : ',argv[0]) # printing first arg, should be data location
    print('model type : ',argv[1]) # priting second arg, should be model you want to predict with
    df = pd.read_csv(argv[0])
    id = pd.DataFrame(df['SK_ID_CURR'])
    df = df.drop('SK_ID_CURR',axis=1)
    to_pred = scale_data(df)
    if argv[1] == 'XGBoost':
        filename = "models/xgboostclassifier.sav"
        loaded_model = pickle.load(open(filename, 'rb'))
        pred = predictions(loaded_model,to_pred)
        id['TARGET'] = pred
        id.to_csv('predictions/xgboost_pred.csv')
        print('Predictions saved in csv')
    elif argv[1] == 'GradientBoostingClassifier':
        filename = "models/gdbclassifier.sav"
        loaded_model = pickle.load(open(filename, 'rb'))
        pred = predictions(loaded_model,to_pred)
        id['TARGET'] = pred
        id.to_csv('predictions/gdbclassifier_pred.csv')
        print('Predictions saved in csv')
    elif argv[1] == 'RandomForest':
        filename = "models/randomforest.sav"
        loaded_model = pickle.load(open(filename, 'rb'))
        pred = predictions(loaded_model,to_pred)
        id['TARGET'] = pred
        id.to_csv('predictions/randomforest_pred.csv')
        print('Predictions saved in csv')
    else:
        print('error')
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])

