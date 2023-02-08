import os
import sys
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def drop_features(df, feature):
    df = df.drop([feature], axis=1, inplace=True)
    return df

class encoding(BaseEstimator, TransformerMixin):
    def __init__(self, column_2=['Vehicle_Age', 'Vehicle_Damage', 'Gender']):
        self.column_2 = column_2
    def fit(self, df):
        return self
    def transform(self, df):
        oe = OrdinalEncoder()
        df[self.column_2] = oe.fit_transform(df[self.column_2])
        return df

class feature_scaling(BaseEstimator, TransformerMixin):
    def __init__(self, column_3=['Annual_Premium']):
        self.column_3 = column_3
    def fit(self, df):
        return self
    def transform(self, df):
        if (set(self.column_3).issubset(df.columns)):
            min_max = MinMaxScaler()
            df[self.column_3] = min_max.fit_transform(df[self.column_3])
            return df
        else:
            print("Error")
            return df
        
class change_dtype(BaseEstimator, TransformerMixin):
    def __init__(self, column_4 = ['Age', 'Driving_License', 'Previously_Insured', 'Vintage']):
        self.column_4 = column_4
    def fit(self, df):
        return self
    def transform(self, df):
        if (set(self.column_4).issubset(df.columns)):
            df[self.column_4] = df[self.column_4].astype('float')
            return df
        else:
            print("Error")

pipe = Pipeline([
    ('encoding', encoding()),
    ('scaler', feature_scaling()),
    ('dtype_conv', change_dtype()),
])

def outlier_thresholds_iqr(df, feature, th1, th3):
    Q1 = df[feature].quantile(th1)
    Q3 = df[feature].quantile(th3)
    IQR = Q3 - Q1
    upper_limit = Q3 + 3 * IQR
    lower_limit = Q1 - 1 * IQR
    return upper_limit, lower_limit

def check_outliers_iqr(df, feature):
    upper_limit, lower_limit = outlier_thresholds_iqr(df, feature, th1=0.05, th3=0.95)
    if df[(df[feature] > upper_limit) | (df[feature] < lower_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds_iqr(df, features, th1=0.05, th3=0.95, replace=True):
    data = []
    for feature in features:
        if feature != 'Response':
            outliers = check_outliers_iqr(df, feature)
            count = None
            upper_limit, lower_limit = outlier_thresholds_iqr(df, feature, th1=0.05, th3=0.95)
            if outliers:
                count = df[(df[feature] > upper_limit) | (df[feature] < lower_limit)][feature].count()
                if replace:
                    if lower_limit < 0:
                        df.loc[(df[feature] > upper_limit), feature] = upper_limit
                    else:
                        df.loc[(df[feature] < lower_limit), feature] = lower_limit
                        df.loc[(df[feature] > upper_limit), feature] = upper_limit
            outliers_status = check_outliers_iqr(df, feature)

def check_outliers_iqr(df, feature):
    upper_limit, lower_limit = outlier_thresholds_iqr(df, feature, th1=0.05, th3=0.95)
    if df[(df[feature] > upper_limit) | (df[feature] < lower_limit)].any(axis=None):
        return True
    else:
        return False



def preprocess_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    drop_features(df, 'id')
    pipe.fit_transform(df)
    replace_with_thresholds_iqr(df, df.columns, th1=0.05, th3=0.95)
    df.to_csv(DATA_PATH[:-4] + "_processed.csv", index=False)

if __name__ == "__main__":
    DATA_PATH = os.path.abspath(sys.argv[1])
    preprocess_data(DATA_PATH)
    print("Saved to {}".format(DATA_PATH[:-4] + "_processed.csv"))
