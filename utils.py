#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    
import time
class timeit(object):
    start_time = 0
    def __init__(self):
        self.start_time = time.time()
    def ptime(self):
        print("- %.4f s -" % (time.time() - self.start_time))
        self.start_time = time.time()
        
from IPython.core.display import display, HTML
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
