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

import model_featutil as mftu
import model_common as mcomm

def main(options, args):
    fTest = args[0]
    fTrain = args[1]
    fCheck = args[2]
    f = open('2010Dow30.txt')
    lsSymTrain = f.read().splitlines(0) + ['$SPX']
    f.close()

    f = open('2010Dow30.txt')
    lsSymTest = f.read().splitlines() + ['$SPX']
    f.close()

   
    lsSym= list(set(lsSymTrain).union(set(lsSymTest)))

    dtStart = dt.datetime(2012, 01, 01)
    dtEnd = dt.datetime(2015, 01, 26)
    #dtStart = dt.datetime(2008, 01, 01)
    #dtEnd = dt.datetime(2008, 02, 28)

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

    lfcFeatures, ldFeatureArgs = mcomm.getFeats()

    ldfTrain = ftu.applyFeatures(dDataTrain, lfcFeatures, ldFeatureArgs, '$SPX')
    print ldfTrain[0]
    ldfTest = ftu.applyFeatures(dDataTest, lfcFeatures, ldFeatureArgs, '$SPX')

    dtStartTrain = dt.datetime(2013, 01, 01)
    dtEndTrain   = dt.datetime(2015, 01, 26)
    dtStartTest  = dt.datetime(2012, 01, 01)
    dtEndTest    = dt.datetime(2012, 12, 31)
    #dtStartTrain = dt.datetime(2008, 01, 01)
    #dtEndTrain   = dt.datetime(2008, 01, 31)
    #dtStartTest  = dt.datetime(2008, 02, 01)
    #dtEndTest    = dt.datetime(2008, 02, 28)

    naTrain = ftu.stackSyms(ldfTrain, dtStartTrain, dtEndTrain) 
    print naTrain.head()
    naTest  = ftu.stackSyms(ldfTest, dtStartTest, dtEndTest) 
    mftu.stackSymsToFile(ldfTrain, fTrain, dtStartTrain, dtEndTrain, bShowRemoved=False)
    mftu.stackSymsToFile(ldfTest, fTest, dtStartTest, dtEndTest, sel="all")
    mftu.stackSymsToFile(ldfTest, fCheck, dtStartTest, dtEndTest, sel="all")

    sys.exit(0)
    ''' Normalize features, use same normalization factors for testing data as training data '''
    ltWeights = ftu.normFeatures( naTrain, -1.0, 1.0, False )
    ''' Normalize query points with same weights that come from test data '''
    ftu.normQuery( naTest[:,:-1], ltWeights )	

    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(naTrain[:,:-1], naTrain[:,-1])

    predicted = model.predict(naTest[:,:-1])
    corrcoef = np.corrcoef(naTest[:,-1], predicted)[0][1]
    print corrcoef
    sys.exit(0)
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

    
def parse_options(paraser): # {{{
    """
    parser command line
    """
    return parser.parse_args()
#}}} 
# execute start here
if __name__ == "__main__": #{{{
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
# }}}
