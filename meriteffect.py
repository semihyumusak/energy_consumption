import random
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk

from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import re
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf

#from tensorflow import keras
#from tensorflow.keras import layers
from datetime import datetime
from datetime import timedelta

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import warnings
import numpy as np
from datetime import date

warnings.filterwarnings('ignore')
import pandas as pd
# from bayes_opt import BayesianOptimization
import xgboost as xgb

from scipy.stats import linregress

from functools import partial
from sklearn.model_selection import StratifiedKFold, KFold
# from tqdm import tqdm, tqdm_notebook
import json
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import svm
# from keras.models import Sequential
# from keras.layers import Activation, Dense

# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def add_datepart(df, fldname, include_month=False):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    if include_month:
        feats = ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start')
    else:
        feats = ('Year', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start')
    for n in feats:
        df[fldname + n] = getattr(fld.dt, n.lower())
    df[fldname + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

# column_name = ["date","hour","price_tl","day_ahead_price_usd","day_ahead_price_eur","production","natural_gas","wind","lignite","stone_coal","imported_coal","fueloil","geothermal","RSH","biomass","RoR","producer_price_index","day_ahead_price_tl","ppi_2011_100","consumption"]
# data = pd.read_csv("/content/drive/My Drive/makaleler/Energy Management/hepsicsv4.csv", names = column_name, na_values = "?", comment = "\t", sep = ";", skipinitialspace = True, decimal=',')
# fileNames = ["tumveri14_6_After_2015.csv"]#,"tumveri14_5.csv"]
# for fileName in fileNames:

# saat bilgisi nasıl eklenebilir?? (one-hot encode)

# 1- wind exclude prediction (production cıkar/cıkarma). exclude edilmeiş pred+exclude edilmiş pred+ exclude ve prod exclude edilmiş pred.
# 2- sentetik veri (diğer parametrelerin orta değerleri ile wind değişkeni)

# print ("-----DESCRIBE ONCESI-----")
# print(d.describe())
# d.describe().to_csv(fileName+"_d_description.csv")
# print ("-----DESCRIBE SONRASI-----")
# # add_datepart(d, "date")
# print ("File: "+fileName)
fileName = "data_new.csv"
path = 'C:\\Users\\semih.yumusak\\Google Drive\\makaleler\\merit effect'
data = pd.read_csv(path+"\\" + fileName, na_values="?", comment="\t", sep=",",
                   skipinitialspace=True, decimal='.')

print(data.columns)


# print(type(data.time[0]))

def prepare_data(d, scenario):
    if (1 in scenario):
        # print(1)
        d["prev_ptf_usd"] = d["ptfusd"].shift(periods=1, fill_value=0)

    if (2 in scenario):
        # print(2)
        # print(d.shape)
        d["prev_pft_usd1"] = d["ptfusd"].shift(periods=1, fill_value=0)
        d["prev_pft_usd2"] = d["ptfusd"].shift(periods=2, fill_value=0)
        d["prev_pft_usd3"] = d["ptfusd"].shift(periods=3, fill_value=0)
        d["prev_pft_usd4"] = d["ptfusd"].shift(periods=4, fill_value=0)
        d["prev_pft_usd5"] = d["ptfusd"].shift(periods=5, fill_value=0)
        # print (d)
        # print(d.shape)
    if (3 in scenario):
        # print(3)
        # print(d.shape)
        d["production0"] = d["production"]
        d["production1"] = d["production"].shift(periods=1, fill_value=0)
        d["production2"] = d["production"].shift(periods=2, fill_value=0)
        d["production3"] = d["production"].shift(periods=3, fill_value=0)
        d["production4"] = d["production"].shift(periods=4, fill_value=0)
        d["production5"] = d["production"].shift(periods=5, fill_value=0)
    # demand geçmişi ekle
    if (4 in scenario):
        d["demand1"] = d["demand"].shift(periods=1, fill_value=0)
        d["demand2"] = d["demand"].shift(periods=2, fill_value=0)
        d["demand3"] = d["demand"].shift(periods=3, fill_value=0)
        d["demand4"] = d["demand"].shift(periods=4, fill_value=0)
        d["demand5"] = d["demand"].shift(periods=5, fill_value=0)
    # 24 saat öncesi
    if (5 in scenario):
        d["demand24"] = d["demand"].shift(periods=24, fill_value=0)
        d["production24"] = d["production"].shift(periods=24, fill_value=0)
        d["prev_pft_usd24"] = d["ptfusd"].shift(periods=24, fill_value=0)

    # time onehotencode
    if (6 in scenario):
        label_encoder = LabelEncoder()
        d['time'] = label_encoder.fit_transform(d['time'])

        one_hot_encoder = OneHotEncoder(sparse=False)

        d = pd.concat((d, pd.DataFrame(one_hot_encoder.fit_transform(d['time'].values.reshape(-1, 1)))), 1)

        # d = d.drop(["date","nafta","other","ptfeur","ptftl","ptfusd"],axis=1)
        # # print(d.columns)

    # production çıkar
    if (7 in scenario):
        d = d.drop(["production"], axis=1)

    d_train = d[~d["date"].str.contains("2020")]
    d_test = d[d["date"].str.contains("2020")]

    y_test = d_test.ptfusd
    y_train = d_train.ptfusd

    d_train = d_train.drop(["date", "time", "nafta", "other", "ptfeur", "ptftl", "ptfusd"], axis=1)
    d_test = d_test.drop(["date", "time", "nafta", "other", "ptfeur", "ptftl", "ptfusd"], axis=1)

    if (8 in scenario):
        d_test["wind"].values[:] = 0
    # print(d["wind"])
    X = d
    # example of training a final regression model

    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(d_train)
    scalarY.fit(y_train.values.reshape(-1, 1))
    X_train = scalarX.transform(d_train)
    y_train = scalarY.transform(y_train.values.reshape(-1, 1))
    X_test = scalarX.transform(d_test)
    y_test = scalarY.transform(y_test.values.reshape(-1, 1))
    # return X, y, scalarX, scalarY, d
    return X_train, X_test, y_train, y_test, scalarX, scalarY, d


from sklearn import linear_model

classifiers = [
    {"name": "SVM SVR", "model": svm.SVR()},
    {"name": "SGDRegressor", "model": linear_model.SGDRegressor()},
    {"name": "BayesianRidge", "model": linear_model.BayesianRidge()},
    {"name": "LassoLars", "model": linear_model.LassoLars()},
    # linear_model.ARDRegression(),
    {"name": "PassiveAggressiveRegressor", "model": linear_model.PassiveAggressiveRegressor()},
    # linear_model.TheilSenRegressor(),
    {"name": "LinearRegression", "model": linear_model.LinearRegression()},
    {"name": "SVM SVR rbf", "model": svm.SVR(kernel='rbf')}]
print("Scenario", end="\t")
for c in classifiers:
    print(c["name"], end="\t")
print("")

# for s in [[0],[1],[2],[3],[4],[5],[6],[1,6],[1,2,3,4,5,6]]:
for s in [[7], [8], [1, 2, 3, 4, 5, 6, 7, 8]]:
    print("Scenario " + str(s), end="\t")
    X_train, X_test, y_train, y_test, scalarX, scalarY, d = prepare_data(data.copy(), s)
    # print(X.shape)
    from sklearn import linear_model

    # print (s, end = "\t")
    for m in classifiers:
        model = m["model"]
        model_name = m["name"]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        y_inv = scalarY.inverse_transform(y_test)
        y_pred_inv = scalarY.inverse_transform(y_pred.reshape(-1, 1))
        rms = mean_squared_error(y_test, y_pred, squared=False)
        print(rms, end="\t")

        d_train = d[~d["date"].str.contains("2020")]
        d_test = d[d["date"].str.contains("2020")]

        fig, ax = plt.subplots()
        ax.plot(d_test.date, y_inv)
        ax.plot(d_test.date, y_pred_inv)
        # print(y_pred_inv.shape)
        # print(d_test.shape)
        folder = "./results/"
        pd.DataFrame(pd.np.column_stack([d_test, y_pred_inv])).to_csv(f'{folder}{s}{model_name}.csv')

        # df2 = pd.DataFrame(y_pred_inv, columns=list('p'))

        # pd.concat([d_test, df2], axis=1).to_csv("deneme")
    print("")
