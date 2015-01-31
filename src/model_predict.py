#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os,sys,shutil
import numpy as np
import math
from math import log
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from time import gmtime, strftime
import scipy
import sys
import sklearn.decomposition
from sklearn.metrics import mean_squared_error
from string import punctuation
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
import time
from scipy import sparse
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import operator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import tree
from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
# parse the command paramaters
from optparse import OptionParser
from model_base import get_date_str


import model_featutil as mftu

import QSTK.qstkfeat.featutil as ftu   

def main(options, args):
    sfTrain = args[0]
    sfTest = args[1]
    sfTestPredict = args[2]
    ftrainY = open(sfTrain, "r")
    ftestY = open(sfTest,"r")
    ftestP = open(sfTestPredict, "w")

    naTrain = mftu.loadFeatFromFile(sfTrain, 2)
    naTest =  mftu.loadFeatFromFile(sfTest, 2)
    ltWeights = ftu.normFeatures( naTrain, -1.0, 1.0, False )
    ftu.normQuery( naTest[:,:], ltWeights )	
    
    print "preparing models"
    model = KNeighborsRegressor(n_neighbors=10)
    selected = [35, 41, 44, 14, 17, 1, 4, 2, 13, 19, 26, 31, 42, 16, 48, 8, 55, 21, 54, 7, 46, 5, 20, 6, 53, 51, 61, 52, 50, 43, 29, 62]
    naTrain = naTrain[:, selected]
    naTest = naTest[:, selected[:-1]]
    model.fit(naTrain[:,:-1], naTrain[:, -1])
    predicted = model.predict(naTest[:,:])
    #corrcoef = np.corrcoef(naTest[:,-1], predicted)[0][1]
    #print naTest[0:10,-1]
    #print predicted[0:10]
    #print corrcoef
 #   tpred = model_predictor.predict(X_test)
 #   score = model_predictor.score(X_test, tpred)
 #   print "score=", score
    assert(len(predicted) == naTest.shape[0])
    lines = open(sfTest).readlines()

    for i in range(len(lines)):
        print >> ftestP, "%s,%f" % (lines[i].strip(),predicted[i])
    sys.exit(0)
    #}}}

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
