
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author 
import sys,os,math
import datetime
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")


''' QSTK Imports '''
import QSTK.qstklearn.kdtknn as kdt
from QSTK.qstkutil import DataAccess as da
from QSTK.qstkutil import qsdateutil as du
from QSTK.qstkutil import tsutil as tsu
 
from QSTK.qstkfeat.features import *
from QSTK.qstkfeat.classes import class_fut_ret

def stackSymsToFile( ldfFeatures, filename, dtStart=None, dtEnd=None, lsSym=None, sDelNan='ALL', bShowRemoved=False ):
    '''
    @summary: Remove symbols from the dataframes, effectively stacking all stocks on top of each other.
    @param ldfFeatures: List of data frames of features.
    @param dtStart: Start time, if None, uses all
    @param dtEnd: End time, if None uses all
    @param lsSym: List of symbols to use, if None, all are used.
    @param sDelNan: Optional, default is ALL: delete any rows with a NaN in it 
                    FEAT: Delete if any of the feature points are NaN, allow NaN classification
                    None: Do not delete any NaN rows
    @return: Numpy array containing all features as columns and all 
    '''
    fout = open(filename , "w")
    
    if dtStart == None:
        dtStart = ldfFeatures[0].index[0]
    if dtEnd == None:
        dtEnd = ldfFeatures[0].index[-1]
    
    naRet = None
    ''' Stack stocks vertically '''
    for sStock in ldfFeatures[0].columns:
        
        if lsSym != None and sStock not in lsSym:
            continue
        naStkData = None
        ''' Loop through all features, stacking columns horizontally '''
        ldfFeatures2 = []
        for dfFeat in ldfFeatures:
            ldfFeatures2.append(dfFeat.ix[dtStart:dtEnd])
        for i in range(0, len(ldfFeatures2[0].index)):
            onelist = [];onelist.append(sStock)
            onelist.append(ldfFeatures2[0].index[i].strftime('%Y-%m-%d'))
            for dfFeat in ldfFeatures2:
                onelist.append(dfFeat.ix[i][sStock])
            if 'ALL' == sDelNan and not math.isnan(np.sum(onelist[2:])) or \
               'FEAT'== sDelNan and not math.isnan(np.sum(onelist[2:-1])):
                for j in range(len(onelist)):
                    print >> fout, onelist[j],
                    if j == len(onelist)-1:
                        print >> fout
                    else:
                        print >> fout, "," ,
            elif bShowRemoved:
                print 'Removed', sStock, str(onelist)


   
