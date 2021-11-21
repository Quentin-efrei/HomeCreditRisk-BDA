import sys
from functions import pd,missing_values,drop_some_columns,fill_some_rows,make_categorical_numerical

def main(argv):
    """
    Test Sphinx
    """
    df = pd.read_csv(argv[0])
    name_to_save = argv[1]
    missing_values_df = missing_values(df)
    df_pourcentage_missing_values = pd.DataFrame({'columns': missing_values_df.index, 'missing percent': missing_values_df.values})
    columns_to_drop = df_pourcentage_missing_values[df_pourcentage_missing_values ['missing percent'] >= 30]['columns'].tolist()
    df = df.drop(columns_to_drop, axis=1)
    df = drop_some_columns(df)
    df = fill_some_rows(df)
    df = df.drop('SK_ID_CURR',axis=1)
    if argv[0] == 'data/application_train.csv':
        temp = df['TARGET']
        df.drop('TARGET',inplace=True,axis=1)
        df = make_categorical_numerical(df)
        df['TARGET'] = temp
        df.to_csv('ready-data/'+name_to_save+'.csv')
    else:
        df = make_categorical_numerical(df)
        df.to_csv('ready-data/'+name_to_save+'.csv')


if __name__ == "__main__":
    main(sys.argv[1:])
