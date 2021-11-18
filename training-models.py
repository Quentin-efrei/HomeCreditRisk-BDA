import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from functions import pd,scale_data
from XGBoost import xgboost

import pickle

def main(argv):
    print(argv)
    df = pd.read_csv(argv[0])
    Y = df['TARGET']
    X = df.drop('TARGET',axis=1)
    X = scale_data(X)
    X_train, X_test, Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.2)
    if argv[1] == 'RandomForest':
        rforest = RandomForestClassifier(n_estimators=20)
        rforest.fit(X_train,Y_train)
        filename = 'models/randomforest.sav'
        pickle.dump(rforest,open(filename,'wb'))
    elif argv[1] == 'XGBoost':
        print('not supported yet')
        xgb = xgboost.XGBClassifier()
        #xgb.fit(X_train,Y_train)
    elif argv[1] == 'GradientBoostig':
        print('not supported yet')
        #gdb = GradientBoostingClassifier()
        #gdb.fit(X_train,Y_train)
    else:
        print('error')
        sys.exit(2)



if __name__ == "__main__":
    # main(['ready-data/train.csv','RandomForest'])
    main(sys.argv[1:])

