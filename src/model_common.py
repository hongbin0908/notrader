
#!/usr/bin/env python
#@author redbin@outlook.com

"""
the common data of the models
"""
import sys,os,json
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from QSTK.qstkfeat.features import *
from QSTK.qstkfeat.classes import class_fut_ret
def getFeats():
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
    return (lfcFeatures, ldFeatureArgs)
