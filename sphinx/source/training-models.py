import sys
from sklearn.model_selection import train_test_split
from functions import pd,scale_data,train_random_forest,predictions,accuracy,train_xgboost,train_gradient_boosting
import mlflow
import mlflow.sklearn

import pickle

def main(argv):
    print('data : ',argv[0])
    print('classifier type : ',argv[1])
    df = pd.read_csv(argv[0])
    Y = df['TARGET']
    X = df.drop('TARGET',axis=1)
    X = scale_data(X)
    X_train, X_test, Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.2)

    mlflow.sklearn.autolog()
    mlflow.xgboost.autolog()

    with mlflow.start_run():



        if argv[1] == 'RandomForest':
            rforest = train_random_forest(X_train,Y_train)
            pred = predictions(rforest,X_test)
            acc = accuracy(pred,Y_test)
            metrics = mlflow.sklearn.eval_and_log_metrics(rforest, X_test, Y_test, prefix="val_")
            mlflow.sklearn.log_model(rforest, "model_random_forest")
            filename = 'models/randomforest.sav'
            pickle.dump(rforest,open(filename,'wb'))
        elif argv[1] == 'XGBoost':
            xgbo = train_xgboost(X_train,Y_train)
            pred = predictions(xgbo,X_test)
            acc = accuracy(pred,Y_test)
            metrics = mlflow.sklearn.eval_and_log_metrics(xgbo, X_test, Y_test, prefix="val_")
            filename = 'models/xgboostclassifier.sav'
            pickle.dump(xgbo,open(filename,'wb'))
        elif argv[1] == 'GradientBoostingClassifier':
            gdb = train_gradient_boosting(X_train,Y_train)
            pred = predictions(gdb,X_test)
            acc = accuracy(pred,Y_test)
            metrics = mlflow.sklearn.eval_and_log_metrics(gdb, X_test, Y_test, prefix="val_")
            filename = 'models/gdbclassifier.sav'
            pickle.dump(gdb,open(filename,'wb'))
        else:
            print('error')
            sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])

