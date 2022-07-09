import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import app as a

def reading(x):
    df=pd.read_csv(x)
    return df,df.columns


def data_type(data):
    dtype=pd.DataFrame(data.dtypes,columns=["DataType"])
    return dtype

def x_n_y(data,var):
    X=data.drop(columns=[str(var)])
    Y=data[var]
    return X,Y

def train_test_split1(X,Y):
    return train_test_split(X, Y, test_size=0.2)

def models(x_train,y_train,x_test,y_test):
    lr_pipeline = Pipeline([('Linear_regression', LinearRegression())])
    DT_pipeline = Pipeline([('DT', DecisionTreeRegressor(random_state=0))])
    RF_pipeline = Pipeline([('RF', RandomForestRegressor(n_estimators=10, random_state=0))])
    SVR_pipeline = Pipeline([('SVR', SVR(kernel='linear'))])
    pipeline_list = [lr_pipeline, DT_pipeline, RF_pipeline, SVR_pipeline]
    for i in pipeline_list:
        i.fit(x_train, y_train)
    lr_pipeline_pred = pd.Series()
    DT_pipeline_pred = pd.Series()
    RF_pipeline_pred = pd.Series()
    SVR_pipeline_pred = pd.Series()
    pipelines_list = [lr_pipeline, DT_pipeline, RF_pipeline, SVR_pipeline]
    pred_list = [lr_pipeline_pred, DT_pipeline_pred, RF_pipeline_pred, SVR_pipeline_pred]
    for i in range(0, len(pipelines_list)):
        pred_list[i] = pipelines_list[i].predict(x_test)
    lr_pipeline_pred = pd.Series(pred_list[0])
    DT_pipeline_pred = pd.Series(pred_list[1])
    RF_pipeline_pred = pd.Series(pred_list[2])
    SVR_pipeline_pred = pd.Series(pred_list[3])
    pred_df = pd.DataFrame()
    pred_df["lr_pipeline_pred"] = lr_pipeline_pred
    pred_df["DT_pipeline_pred"] = DT_pipeline_pred
    pred_df["RF_pipeline_pred"] = RF_pipeline_pred
    pred_df["SVR_pipeline_pred"] = SVR_pipeline_pred
    pred_df["y_test"] = y_test.values
    RMSE_df = pd.DataFrame()
    for i in pred_df.columns:
        RMSE_df[i + " RMSE"] = pd.Series(mean_squared_error(pred_df[i], pred_df["y_test"]) ** (0.5))
    #model_names=["Linear Regression","Decison Tree","Random Forest","SVR"]
    plt.figure(figsize=(13, 6))
    plt.bar(RMSE_df.T.index,RMSE_df.T[0])
    plt.savefig('static/regression.png')
    return RMSE_df.T



def preprocessing(x):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
    le = LabelEncoder()
    dtype = pd.DataFrame(x.dtypes, columns=["DataType"])
    cat_var = dtype[dtype.DataType == "object"].index.to_list()
    for i in cat_var:
        label_encoding = le.fit_transform(x[i])
        x[i + "_label_encoding"] = label_encoding
    x = x.drop(columns=cat_var)
    for i in list(x.columns):
        if x[i].isnull().sum() > 0:
            x[i] = x[i].fillna(x[i].mean())

    def remove_outlier(x):
        sorted(x)
        Q1, Q3 = np.percentile(x, [25, 75])
        IQR = Q3 - Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        return lower_range, upper_range

    for i in range(0, len(x.columns)):
        lratio, uratio = remove_outlier(x.iloc[:, i])
        x.iloc[:, i] = np.where(x.iloc[:, i] > uratio, uratio, x.iloc[:, i])
        x.iloc[:, i] = np.where(x.iloc[:, i] < lratio, lratio, x.iloc[:, i])
    x = x.apply(zscore)


    return x
