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
import sys, time, json

import numpy as np
import pandas as pand


import QSTK.qstkfeat.featutil as ftu   

import model_featutil as mftu
import model_common as mcomm

from functions import sequentialForwardSelection

def main(options, args):
    fTrain = args[0]
    fTest = args[1]
    fSelectedFeats = args[2]
    
    fdSelectedFeats = open(fSelectedFeats, "w")

    naTrain = mftu.loadFeatFromFile(fTrain, 2)
    naTest = mftu.loadFeatFromFile(fTest,2)


    ltWeights = ftu.normFeatures(naTrain, -1.0, 1.0, False)
    ftu.normQuery(naTest[:,:-1], ltWeights)
    
    lfcFeatures, ldFeatureArgs = mcomm.getFeats()

    lFeatures = range(0, len(lfcFeatures)-1)
    classLabelIndex = len(lfcFeatures) -1

    maxlCorrCoef, selectedFeats = sequentialForwardSelection(naTrain, naTest, lFeatures, classLabelIndex)
    print >> fdSelectedFeats, json.dumps(selectedFeats) 
    
def parse_options(paraser): # {{{
    """
    parser command line
    """
    parser.add_option("--input", dest="input",action = "store", default=None, help = "the input filename dir")
    parser.add_option("--short", dest="short",action = "store", default=-1, help = "using short data")
    parser.add_option("--utildate", dest="utildate",action = "store", default=None, help = "the last date to train")
    return parser.parse_args()
#}}} 

# execute start here
if __name__ == "__main__": #{{{
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
# }}}
