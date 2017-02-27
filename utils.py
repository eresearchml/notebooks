#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
import time
import scipy as sp
import seaborn as sns; sns.set()
import itertools
def boxplotcats(df, c2, catcols):
    for c in catcols:
        if(len(onehotencodings[c]) < 100):
            #df_sub = inverseonehotencode(df, c,onehotencodings[c])
            _ = sns.boxplot(x=c, y=c2, data=df)
            _ = plt.xticks(rotation=90)
            plt.show()

def inverseonehotencode(df, col, cols):
    df2 = df.copy(deep=True)
    x = df2[cols].stack()
    s = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))
    df2[col] = s
    df2 = df2.drop(cols, axis=1)
    #print(s)
    return df2

def plot_catmatrix(cm, x_labels, y_labels,
                          normalize=False,
                          title='matrix',
                          cmap=plt.cm.Blues,
                          y_title="", x_title="",
                          axis=0):
   
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(x_labels))
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.grid(False) 
    #for axi in (ax.xaxis, ax.yaxis):
    #    for tic in axi.get_major_ticks():
    #        tic.tick1On = tic.tick2On = True
    #        tic.label1On = tic.label2On = False
    orig = cm
    if normalize:
        if axis == 1:
            cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
        else:
            cm = cm.astype('float') / cm.sum(axis=axis)
    thresh = (cm.max() - cm.min()) * 0.5 + cm.min()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v = cm[i,j]
        if(np.isnan(v)):
            s = '-'
        else:
            s = "{:1.2f}".format(v)
        #s = "{:1.4f}".format(cm[i,j]) + "("+str(orig[i,j])+")"
        plt.text(j, i, s, horizontalalignment="center",fontsize=8,
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.show()

def vc(df, c1, num=30):
    vc1 = df[c1].value_counts()
    n1 = vc1.index.tolist()
    
    if(len(n1) > num):
        df_temp = filteronsig(df, [c1], num=num)
        vc1 = df_temp[c1].value_counts()
        n1 = vc1.index.tolist()
    return vc1, n1


            
def plotcols(df, c1, c2):
    print(c1, c2)
    if(c1 in df.columns):
        vc1, n1 = vc(df, c1)
    else:
        return
    if(c2 in df.columns):
        vc2, n2 = vc(df, c2)
    else:
        return
    
    counts = np.empty((len(n1),len(n2),))
    counts[:] = np.NAN
    #print(vc2)
    for i in range(len(n1)):
        for j in range(len(n2)):
            if(c1 in df.columns and c2 in df.columns):
                c = len(df[(df[c1]==n1[i]) & (df[c2]==n2[j])])
                #print(c)
                counts[i,j] = c
    #print(counts)
    plot_catmatrix(np.transpose(counts), n1, n2, normalize=True, y_title=c2, x_title=c1)
    #x = LabelEncoder().fit_transform(df[c1])
    #y = LabelEncoder().fit_transform(df[c2])
    #plt.plot(x, y, 'o')
    #plt.show()



def plot_matrix(cm, classesx, classesy,
                          normalize=True,
                          title='matrix',
                          cmap=plt.cm.Blues,
                          axis=0):
   
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marksx = np.arange(len(classesx))
    plt.xticks(tick_marksx, classesx, rotation=90)
    tick_marksy = np.arange(len(classesy))
    plt.yticks(tick_marksy, classesy)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.grid(False) 
    #for axi in (ax.xaxis, ax.yaxis):
    #    for tic in axi.get_major_ticks():
    #        tic.tick1On = tic.tick2On = True
    #        tic.label1On = tic.label2On = False
    orig = cm
    if normalize:
        if axis == 1:
            cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
        else:
            cm = cm.astype('float') / cm.sum(axis=axis)
    thresh = (cm.max() - cm.min()) * 0.5 + cm.min()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        s = "{:1.2f}".format(cm[i,j])
        #s = "{:1.4f}".format(cm[i,j]) + "("+str(orig[i,j])+")"
        plt.text(j, i, s, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('Y label')
    plt.xlabel('X label')

def onelr(x1, y1):
    lr = LinearRegression()
    lr.fit(x1, y1)
    
    return lr.score(x1, y1)

def plotcorrel(df_temp, colsx, colsy):
    num_coef = np.empty((len(colsx),len(colsy),))
    num_coef[:] = np.NAN
    x_labels = []
    y_labels = []
    for i in range(len(colsx)):
        c_i = colsx[i]
        if(c_i in df_temp.columns):
            df_temp = df_temp.dropna(subset=[c_i])
            y_labels.append(c_i + "("+str(df_temp.shape[0])+")")
            #print(c_i, df_temp.shape)
            for j in range(len(colsy)):
                c_j = colsy[j]
                if len(x_labels)  < len(colsy):
                     x_labels.append(c_j + "("+str(df_temp.shape[0])+")")
                if(c_j in df_temp.columns):
                    df_temp = df_temp.dropna(subset=[c_j])
                    vals_i = df_temp[c_i].values.reshape(-1,1)
                    vals_j = df_temp[c_j].values.reshape(-1,1)
                    if(np.isnan(vals_j).sum() == 0):
                        #r = onelr(vals_i, vals_j)
                        pcor = sp.stats.pearsonr(vals_i, vals_j)
                        #print(pcor)
                        num_coef[i, j] = pcor[0][0]
    #print(num_coef)
    figurefullwidth()
    plot_matrix(num_coef, x_labels, y_labels, normalize=False, title="")

def onehotencode(df, cols):
    onehotencodings = {}
    for c in cols:
        if(c in df.columns):
            pre_cols = df.columns
            temp_col = df[c]
            df = pd.get_dummies(df, columns=[c])
            df[c] = temp_col
            post_cols = df.columns
            diff_cols = list(set(post_cols) - set(pre_cols))
            onehotencodings[c] = diff_cols
    return df, onehotencodings

def encodecols(df, catcols, dummy=False):
    df2 = df.copy(deep=True)
    df2 = onehotencode(df2, catcols)
    return df2

def printencodings(catcols):
    tenc1 = pd.DataFrame()
    for c in catcols:
        if(c in onehotencodings):
            classes = onehotencodings[c]
            tenc = pd.DataFrame(classes, index = range(len(classes)))
            tenc[c] = tenc[0]
            del tenc[0]
            dispdf(tenc, num=10)

def dispcatdata(df, c):
    vc = df[c].value_counts()
    print(c, " counttypes ",vc.shape[0],  "\n")
    if(vc.shape[0] > 100):
        print("Data is large!!!!!!!!!!!! " , vc.shape[0])
    cumsum = np.cumsum(vc.values)
    cumsum = cumsum/len(df[c])
    if False:
        subplottitle(1, "CumSum")
        plt.plot(cumsum)
    #print(vc)
    figurefullwidth()
    vc[:100].plot(kind='bar')
    plt.show()
    pro = 0;
    for n in vc.values:
        pro
    
def sigvals(df, c, threshold=0.8, num=0):
    print(df.columns, c)
    vc = df[c].value_counts()
    names = vc.index.tolist()
   
    count = len(df[c])
    cutoff = (int)(count * threshold)
    acc = 0
    num_acc = 0;
    ret = []
    others = []
    for i in range(vc.shape[0]):
        #print(names[i])
        if acc < cutoff and num_acc < num -1:
            ret.append(names[i])
        else:
            others.append(names[i])
        acc += vc[i]
        num_acc += 1
    return ret, others


def filteronsig(df, cols, threshold=0.8, num=0):
    df2 = df.copy(deep=True)
    for c in cols:
        #print(c)
        sig, others = sigvals(df2, c, threshold=threshold, num=num)
        df2[c] = df2[c].apply(lambda x: x if x in sig else "other")
    return df2

def examineonecatcol(df_local, c, threshold):
    df_temp = filteronsig(df_local, [c], threshold=threshold)
    diff = df_local.shape[0] - df_temp.shape[0]
    print(df_temp.shape, diff, "{:1.2f}".format(diff/df_local.shape[0]*100), "% elements in bottom ", "{:1.2f}".format((1-threshold)*100), "% of categories")
    dispcatdata(df_local, c)
    
    #dist functions
def showdist(df, c, threshold=3.5):
    figurefullwidth()
    nans = df[c].isnull().sum()
    vals = df[c].dropna().values
    subplottitle(1, "Nans", w=6)
    plt.bar([1, 2], [len(df[c])-nans,nans], tick_label = ["Ok", "NaN"])
    subplottitle(2, "Hist", w=6)
    plt.hist(vals, bins=20)
    subplottitle(3, "Plot", w=6)
    plt.plot(vals, 'o')
    subplottitle(4, "Box", w=6)
    sns.boxplot(vals)
    outliers = mad_based_outlier(df[c].values)
    outpoints = df[c][outliers]
    subplottitle(5, "Distplot", w=6)
    sns.distplot(df[c], bins=20, hist=False)
    subplottitle(6, "Outliers", w=6)
    plt.plot(outpoints, 'ro')
    out_per = (len(outpoints)/df.shape[0])*100
    nan_per = (nans/df.shape[0])*100
    print(c,df.shape,"Ouliers", len(outpoints), "{:1.2f}".format(out_per),"%","Nans",nans, "{:1.2f}".format(nan_per),"%","\n")
    plt.show()


def examineonenumcol(df, c):
    showdist(df, c)
    #df2 = removeoutliers(df, c)
    #chg = df2.shape[0] - df.shape[0]
    #chg_per = chg / df.shape[0] * 100.0
    #print(c, "After",df2.shape,chg, "{:10.2f}".format(chg_per) + "%", "\n")
    #showdist(df2, c)

def removeoutliers(df, c, threshold=3.5):
    outliers = mad_based_outlier(df[c].values)
    outpoints = df[c][outliers]
    return df.drop(outpoints.index)

def removeoutliersfromcols(df, cols, threshold=3.5):
    print("#===============================================")
    print("# Remove outliers from", cols)
    print("#===============================================")

    df2 = df.copy(deep=True)

    for c in cols:
        if(c in df2.columns):
            df2 = removeoutliers(df2, c)
    return df2

def removecols(df, cols):
    print("#===============================================")
    print("# Remove cols ", cols)
    print("#===============================================")
    df2 = df.copy(deep=True)

    for c in cols:
        if(c in df2.columns):
            df2 = df2.drop(c, axis=1)
    return df2
def removenansfromcols(df, cols):
    print("#===============================================")
    print("# Remove nan from cols ", cols)
    print("#===============================================")
    df2 = df.copy(deep=True)

    for c in cols:
        if(c in df2.columns):
            val_list = df2[df2[c].apply(lambda x: np.isnan(x))]
            print(len(val_list))
            df2 =df2.drop(val_list.index)
    return df2
def makenormalfromexp(df, cols):
    print("#===============================================")
    print("# make normal from exp", cols)
    print("#===============================================")
    df2 = df.copy(deep=True)

    for c in cols:
        if(c in df2.columns):
            df2[c] = df2[c].apply(lambda x: 1/x)
    return df2
def makenormalfromlog(df, cols):
    print("#===============================================")
    print("# make normal from lognormal", cols)
    print("#===============================================")
    df2 = df.copy(deep=True)

    for c in cols:
        if(c in df2.columns):
            df2[c] = df2[c].apply(lambda x: np.log(x))
    return df2

def splitcoltypes(df):
    numcols = []
    datcols = []
    catcols = []
    for i in range(len(df.columns)):
        if(df.dtypes[i] == "object"):
            catcols.append(df.columns[i])
        elif(df.dtypes[i] == "datetime64[ns]"):
            datcols.append(df.columns[i])
        else:
            numcols.append(df.columns[i])
    return numcols, datcols, catcols
def coltodatetime(df, col):
    df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
def splitcol(df, col, char):
    df[col] =df[col].fillna('')
    r = df[col].apply(lambda x: pd.Series(x.split(char)))
    for i in range(0, r.shape[1]):
        df[col + str(i)] = r[i]
    #del df[col]
def combinedatentime(df, datecol, timecol, newcol):
    df[newcol] = df.apply(lambda row: datetime.datetime.combine(row[datecol].date(), row[timecol].time()), axis=1) 
def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def colstats(df, c):
    vals = df[c]
    count = len(vals)
    min = np.min(vals)
    max = np.max(vals)
    per_25 = np.percentile(vals, 25)
    mean = np.mean(vals)
    per_75 = np.percentile(vals, 75)
    std = np.std(vals)
    #expected 3
    kurt = sp.stats.kurtosis(vals)
    skew = sp.stats.skew(vals)
    nans = vals.isnull().sum()
    per_nans = nans/count * 100
    out_num = len(df[c][mad_based_outlier(df[c].values)])
    out_per = out_num/count * 100

    d = {"count": count, "min": min, "max": max, "per_25": per_25, "mean": mean,
         "per_75": per_75, "std": std, "kurt": kurt, "skew": skew, "nans": nans, 
         "nans_per": per_nans, "out_num": out_num, "out_per": out_per
        }
    
    st = pd.DataFrame.from_dict(d, orient='index')
    st.rename(columns={0: c }, inplace=True)
    st.sort_index(inplace=True)
    return st

def catstats(df, c):
    vc = df[c].value_counts()
    count = len(vc)

    d = {"catcount": count, 
        }
    
    st = pd.DataFrame.from_dict(d, orient='index')
    st.rename(columns={0: c }, inplace=True)
    st.sort_index(inplace=True)
    return st


def removenans(df, c):
    return df.dropna(subset=[c])

def dataframestats(df):
    stats = pd.DataFrame()
    for c in df.columns:
        if(not df[c].dtype == "object" and not df[c].dtype == "datetime64[ns]"):
            st = colstats(df, c)
            stats[c] = st[c]
    return stats

def dataframecatstats(df_local):
    stats = pd.DataFrame()
    for c in df_local.columns:
        if(not df_local[c].dtype == 'float64' and not df_local[c].dtype == 'int64'):
            print(df_local.shape)
            st = catstats(df_local, c)
            stats[c] = st[c]
    return stats
'standard utils for delliott'
def init():
    import warnings
    warnings.filterwarnings('ignore')
    np.set_printoptions(linewidth=150)
    
    #Tracer()()
def prep (a):
    return a.reshape(a.shape[0], 1)
def unprep (a):
    return np.squeeze(a)
def figurefullwidth():
    plt.figure(figsize=(25,5));
def subplot(n):
    plt.subplot(1,3,n);
def subplottitle(n, title, w=3):
    ax = plt.subplot(1, w, n)
    ax.set_title(title)
    

class timeit(object):
    start_time = 0
    def __init__(self):
        self.start_time = time.time()
    def ptime(self):
        print("- %.4f s -" % (time.time() - self.start_time))
        self.start_time = time.time()
        

def dispdf(df, width=10, num=0, cols=0):
    
    if cols==0:
        cols=len(df.columns)
    if num == 0:
        if(df.shape[1] > 20):
            num = 5
        else:
            num = df.shape[1]
        
    #print(list(range(0, len(df.columns), width)))
    for i in range(0, cols, width):
        #print(df.ix[:,0:5].head())
        display(df.ix[:,i:i+min(width, len(df.columns))].head(num))
    
        
def encodeonehot(df, cols):
    
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df
def printfull(x):
    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')
