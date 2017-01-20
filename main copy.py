import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import statsmodels as sm

#---------------
# read in data and trim
f1 = "/Users/david/notebooks/data/futs1_errors.csv"
f2 = "/Users/david/notebooks/data/futs2_errors.csv"
df = pd.read_csv(f2, header=3, nrows=150, infer_datetime_format=True)
df = df.convert_objects(convert_numeric=True)
#convert to datetime
#df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
#pick columns
df = df[df.columns[:3]]
#pick rows
df.iloc[4:30]
#eyeball
print(df.head())

#----------------
# examine data columns
#min, max, count
print(df.describe())
print(df.dtypes)
#look for rogue data, nas etc
# filter to see if nas or inf or -veinf
nans = df[df["OPEN"].apply(np.isnan)]
nans = nans.append(df[df["VOLUME"].apply(np.isnan)])
df = df.drop(nans.index)
#examine data shape
#look at the value counts - best for cats
g1 = df["OPEN"].value_counts()
print(g1)
#histogram data
#plt.hist(df["OPEN"], bins=20)
#plt.show()
#plt.plot(df["OPEN"])
#plt.show()
print(df.describe())
#plt.figure()
#plt.boxplot(df["OPEN"])
#plt.show()

#examine outliers
def outlier(points, threshold=2):
    mean = np.mean(points)
    stdev = np.std(points)
    zscores = np.abs((points - mean)/stdev)
    return zscores > threshold

out1 = outlier(df["VOLUME"].values)
df = df.drop(df[out1].index)

#reindex dataframe
df = df.reset_index(drop=True)
#print(df)

#dummy encoding
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#df["something"] = le.fit_transform(df["something"])

#show the important time series data
#plt.figure()
#plt.plot(df["OPEN"])
#plt.show()
df["CHG0"] = df["OPEN"] - df["OPEN"].shift(1)
for i in range(1,13):
    df["CHG"+str(i)] = df["CHG0"].shift(i)
print(df.head(15))
df = df.drop(range(0,13))
#print(df.head(20))
#plt.figure()
#plt.acorr(df["CHG"])
#plt.show()

def splitdata(points, split):
    ret = []
    cutoff = 0
    while (cutoff < len(points)):
        cutoff += len(points) * split
        ret.append(points[:cutoff])

    return ret, points[cutoff:]
from sklearn.linear_model import LinearRegression
def plot(i, j):
    lr = LinearRegression()
    x = df["CHG"+str(i)].reshape(-1, 1)
    y = df["CHG"+str(j)].reshape(-1,1)
    lr.fit(x, y)
    pred = lr.predict(x)
    plt.figure()
    plt.scatter(x, y)
    plt.scatter(x, pred, c='r')
    plt.show()

#plot(1,2)
#plot(1,12)

feat_list = ["CHG"+str(i) for i in range(1, 13)]

#partition into test and train
split = 0.8
cutoff = (int)(df.shape[0] * split)
#r, t = splitdata(df["CHG"].values, 0.8)
x1 = df[feat_list].values[:cutoff]
y1 = df["CHG0"].values[:cutoff]
x2 = df[feat_list].values[cutoff:]
y2 = df["CHG0"].values[cutoff:]
print(df.shape, x1.shape, y1.shape, x2.shape, y2.shape)

#x1 = test[:-1].reshape(-1,1)
#y1 = test[1:].reshape(-1,1)
#x2 = train[:-1].reshape(-1,1)
#y2 = train[1:].reshape(-1,1)

#print(x1, y1)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x1, y1)
y2_pred = lr.predict(x2)

def error(pred, act):
    diff = np.mean(np.power(pred-act, 2))
    return diff
plt.plot(df["CHG0"])
plt.show()
print("var", lr.score(x2,y2))
print("mse", error(y2_pred, y2))
print("coef", lr.coef_)



