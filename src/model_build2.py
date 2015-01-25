#!/usr/bin/env python
#@author redbin@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os,json
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
# parse the command paramaters
from optparse import OptionParser
import sys
import time

import numpy as np
import pandas as pand
import matplotlib.pyplot as plt

''' QSTK imports '''
from QSTK.qstkutil import DataAccess as da
from QSTK.qstkutil import qsdateutil as du

from QSTK.qstkfeat.features import *
from QSTK.qstkfeat.classes import class_fut_ret
import QSTK.qstkfeat.featutil as ftu   


from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from functions import sequentialForwardSelection

def main(options, args):
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    print y_true.shape
    print roc_auc_score(y_true, y_scores)
    f = open('2008Dow30.txt')
    lsSymTrain = f.read().splitlines() + ['$SPX']
    f.close()

    f = open('2010Dow30.txt')
    lsSymTest = f.read().splitlines() + ['$SPX']
    f.close()

   
    lsSym= list(set(lsSymTrain).union(set(lsSymTest)))

    dtStart = dt.datetime(2008, 01, 01)
    dtEnd = dt.datetime(2010, 12, 31)

    norObj = da.DataAccess('Yahoo')
    ldtTimestamps = du.getNYSEdays(dtStart, dtEnd, dt.timedelta(hours = 16))

    lsKeys = ['open', 'high', 'low', 'close', 'volume']

    ldfDataTrain = norObj.get_data(ldtTimestamps, lsSymTrain, lsKeys)
    ldfDataTest =  norObj.get_data(ldtTimestamps, lsSymTest, lsKeys)
    for temp in ldfDataTrain:
        temp.fillna(method = "ffill").fillna(method="bfill")


    for temp in ldfDataTest:
        temp.fillna(method="ffill").fillna(method="bfill")

    dDataTrain = dict(zip(lsKeys, ldfDataTrain))

    dDataTest = dict(zip(lsKeys, ldfDataTest))


    lfcFeatures = [ featMA, featMA, featMA, featMA, featMA, featMA, \
               featRSI, featRSI, featRSI, featRSI, featRSI, featRSI, \
               featDrawDown, featDrawDown, featDrawDown, featDrawDown, featDrawDown, featDrawDown, \
               featRunUp, featRunUp, featRunUp, featRunUp, featRunUp, featRunUp, \
               featVolumeDelta, featVolumeDelta, featVolumeDelta, featVolumeDelta, featVolumeDelta, featVolumeDelta, \
               featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, featAroon, \
               #featStochastic, featStochastic, featStochastic, featStochastic, featStochastic, featStochastic,featStochastic, featStochastic, featStochastic, featStochastic, featStochastic, featStochastic, \
               featBeta, featBeta, featBeta, featBeta, featBeta, featBeta,\
               featBollinger, featBollinger, featBollinger, featBollinger, featBollinger, featBollinger,\
               featCorrelation, featCorrelation, featCorrelation, featCorrelation, featCorrelation, featCorrelation,\
               featPrice, \
               featVolume, \
               class_fut_ret \
               ]

    ldFeatureArgs = [  {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5,'bDown':True},{'lLookback':10,'bDown':True},{'lLookback':20,'bDown':True},{'lLookback':5,'bDown':False},{'lLookback':10,'bDown':False},{'lLookback':20,'bDown':False},{'lLookback':5,'bDown':True,'MR':True},{'lLookback':10,'bDown':True,'MR':True},{'lLookback':20,'bDown':True,'MR':True},{'lLookback':5,'bDown':False,'MR':True},{'lLookback':10,'bDown':False,'MR':True},{'lLookback':20,'bDown':False,'MR':True},\
            #{'lLookback':5,'bFast':True},{'lLookback':10,'bFast':True},{'lLookback':20,'bFast':True},{'lLookback':5,'bFast':False},{'lLookback':10,'bFast':False},{'lLookback':20,'bFast':False},{'lLookback':5,'bFast':True,'MR':True},{'lLookback':10,'bFast':True,'MR':True},{'lLookback':20,'bFast':True,'MR':True},{'lLookback':5,'bFast':False,'MR':True},{'lLookback':10,'bFast':False,'MR':True},{'lLookback':20,'bFast':False,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {'lLookback':5},{'lLookback':10},{'lLookback':20}, {'lLookback':5,'MR':True},{'lLookback':10,'MR':True},{'lLookback':20,'MR':True},\
            {},\
            {}, \
            {'i_lookforward':5} \
            ]
    ldfTrain = ftu.applyFeatures(dDataTrain, lfcFeatures, ldFeatureArgs, '$SPX')
    ldfTest = ftu.applyFeatures(dDataTest, lfcFeatures, ldFeatureArgs, '$SPX')

    dtStartTrain = dt.datetime(2008, 01, 01)
    dtEndTrain   = dt.datetime(2009, 12, 31)
    dtStartTest  = dt.datetime(2010, 01, 01)
    dtEndTest    = dt.datetime(2010, 12, 31)

    naTrain = ftu.stackSyms(ldfTrain, dtStartTrain, dtEndTrain) 
    naTest  = ftu.stackSyms(ldfTest, dtStartTest, dtEndTest) 
    ''' Normalize features, use same normalization factors for testing data as training data '''
    ltWeights = ftu.normFeatures( naTrain, -1.0, 1.0, False )
    ''' Normalize query points with same weights that come from test data '''
    ftu.normQuery( naTest[:,:-1], ltWeights )	
    naTrainFeats = naTrain[:,:-1]
    naTrainClasses = naTrain[:,-1]
    naTestFeats = naTest[:,:-1]
    naTestClasses = naTest[:,-1]

    print "naTrainFeats:", naTrainFeats.shape, "naTrainClasses:", naTrainClasses

    cLearn = ftu.createKnnLearner(naTrain, lKnn=5)
    Ypredicted = cLearn.query(naTestFeats)
    corrcoef = np.corrcoef(naTestClasses, Ypredicted)[0][1]
    print corrcoef
    #model = GradientBoostingRegressor(max_features=1.0, learning_rate=0.1, max_depth=5, n_estimators=300)
    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(naTrainFeats, naTrainClasses)

    predicted = model.predict(naTestFeats)
    print naTestClasses
    print predicted
    from sklearn.metrics import r2_score

    print r2_score(naTestClasses, predicted)
    
    corrcoef = np.corrcoef(naTestClasses,predicted)[0][1]
    print corrcoef

    lFeatures = range(0, len(lfcFeatures)-1)
    classLabelIndex = len(lfcFeatures)-1
    sequentialForwardSelection(naTrain, naTest, lFeatures, classLabelIndex)

if __name__ == "__main__":
    main(None, None)
    
