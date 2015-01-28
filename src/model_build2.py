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
    fTrain = args[0]
    sDateStart = args[1]
    sDateEnd = args[2]
    f = open('src/2010Dow30.txt')
    lsSym = f.read().splitlines(0) + ['$SPX']
    f.close()

   
    dtStart = dt.datetime.strptime(sDateStart, '%Y-%m-%d') 
    dtEnd = dt.datetime.strptime(sDateEnd, '%Y-%m-%d')

    norObj = da.DataAccess('Yahoo')
    ldtTimestamps = du.getNYSEdays(dt.datetime.strptime("2010-01-01", '%Y-%m-%d'), dtEnd, dt.timedelta(hours = 16))
    print ldtTimestamps

    lsKeys = ['open', 'high', 'low', 'close', 'volume']

    ldfData = norObj.get_data(ldtTimestamps, lsSym, lsKeys)
    print ldfData[0]
    for temp in ldfData:
        temp.fillna(method = "ffill").fillna(method="bfill")

    dData = dict(zip(lsKeys, ldfData))

    lfcFeatures, ldFeatureArgs = mcomm.getFeats()

    ldfTrain = ftu.applyFeatures(dData, lfcFeatures, ldFeatureArgs, '$SPX')
    #print ldfTrain[0]

    naTrain = ftu.stackSyms(ldfTrain, dtStart, dtEnd) 
    print naTrain
    mftu.stackSymsToFile(ldfTrain, fTrain, dtStart, dtEnd, bShowRemoved=True)
    
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
