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


import model_featutil as mftu
def main(options, args):
    lsSymTrain = ['AA']+ ['$SPX']

   

    dtStart = dt.datetime(2008, 01, 01)
    dtEnd = dt.datetime(2008, 01,30) 

    norObj = da.DataAccess('Yahoo')
    ldtTimestamps = du.getNYSEdays(dtStart, dtEnd, dt.timedelta(hours = 16))

    lsKeys = ['open', 'high', 'low', 'close', 'volume']

    ldfDataTrain = norObj.get_data(ldtTimestamps, lsSymTrain, lsKeys)
    for temp in ldfDataTrain:
        temp.fillna(method = "ffill").fillna(method="bfill")

    dDataTrain = dict(zip(lsKeys, ldfDataTrain))


    lfcFeatures = [ featMA,\
               class_fut_ret \
               ]

    ldFeatureArgs = [  {'lLookback':5},\
            {'i_lookforward':5} \
            ]
    ldfTrain = ftu.applyFeatures(dDataTrain, lfcFeatures, ldFeatureArgs, '$SPX')
    print ldfTrain

    dtStartTrain = dt.datetime(2008, 01, 01)
    dtEndTrain   = dt.datetime(2008, 01, 30)

    naTrain = ftu.stackSyms(ldfTrain, dtStartTrain, dtEndTrain) 
    mftu.stackSymsToFile(ldfTrain, "/dev/stdout", dtStartTrain, dtEndTrain) 
    print naTrain
    naTest = naTrain.copy()
    np.savetxt("naTrain_org.csv", naTrain)
    ''' Normalize features, use same normalization factors for testing data as training data '''
    ltWeights = ftu.normFeatures( naTrain, -1.0, 1.0, False )
    print ltWeights
    ''' Normalize query points with same weights that come from test data '''
    ftu.normQuery( naTest[:,:-1], ltWeights )	

    np.savetxt("naTrain.csv", naTrain)
    np.savetxt("naTest.csv", naTest)


if __name__ == "__main__":
    main(None, None)
    
