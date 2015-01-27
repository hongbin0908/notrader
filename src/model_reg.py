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


import QSTK.qstkfeat.featutil as ftu   

def main(options, args):
    sfTrain = args[0]
    sfTest = args[1]
    sfTestPredict = args[2]
    ftrainY = open(sfTrain, "r")
    ftestY = open(sfTest,"r")
    ftestP = open(sfTestPredict, "w")
    l_X = []

    for line in ftrainY:
        tokens = line.split(",")
        features = []
        for i in range(len(tokens)):
            if i < 2:
                continue
            else:
                features.append(float(tokens[i]))
        l_X.append(features)
    X_train = np.array(l_X)
    if int(options.short) > 0:
        print "using short data for test purpose"
        X_train = X_train[0:int(options.short)]

    l_X = []
    for line in ftestY:
        tokens = line.split(",")
        features = []
        for i in range(len(tokens)):
            if i < 2 :
                continue
            else:
                features.append(float(tokens[i]))
        l_X.append(features)
    X_test = np.array(l_X)



    ltWeights = ftu.normFeatures( X_train, -1.0, 1.0, False )
    ''' Normalize query points with same weights that come from test data '''
    ftu.normQuery( X_test[:,:-1], ltWeights )	
    
    print "preparing models"
    model = KNeighborsRegressor(n_neighbors=10)
    selected = [47, 57, 38, 19, 49, 4, 18, 61, 16, 10, 20, 37, 34, 62]
    X_train = X_train[:, selected]
    X_test = X_test[:, selected]
    model.fit(X_train[:,:-1], X_train[:, -1])
    predicted = model.predict(X_test[:,:-1])
    corrcoef = np.corrcoef(X_test[:,-1], predicted)[0][1]
    print corrcoef
#    tpred = model_predictor.predict(X_test)
 #   score = model_predictor.score(X_test, tpred)
 #   print "score=", score
    assert(len(predicted) == X_test.shape[0])
    lines = open(sfTest).readlines()

    for i in range(len(lines)):
        print >> ftestP, "%s,%f" % (lines[i].strip(),predicted[i])
    sys.exit(0)
    #{{{ prediction
    print "prediction ..."
    stock_predict_out = file(options.input + "/" + options.utildate + "/predict.csv", "w")
    for line in file(options.input + "/" + options.utildate + "/last.csv", "r"):
        tokens = line.split(",")
        l_features = []
        for i in range(len(tokens)):
            if 0 == i:
                print >> stock_predict_out, "%s," % tokens[i],
            elif 1 == i:
                print >> stock_predict_out, "%s,1," % tokens[i],
            else:
                l_features.append(float(tokens[i].strip()))
        l_features2 = []
        l_features2.append(l_features)
        np_features = np.array(l_features2)
        if np_features.shape[1] != X_test.shape[1] :
            assert(false)
        pred = model_predictor.predict_proba(np_features)
        print >> stock_predict_out, "%f" % pred[0,1]
    stock_predict_out.close()
    
    while True:
        if not os.path.exists("../data/model_tunner.list"):
            print ".",
            time.sleep(1)
            continue 
        for dirname in open("../data/model_tunner.list","r"):
            dirname = dirname.strip()
            print dirname
            if not os.path.exists(dirname):
                print "%s not exists" % dirname
                continue
            stock_predict_out = file(dirname + "/predict.csv", "w")
            for line in file(dirname + "/last.csv", "r"):
                tokens = line.split(",")
                l_features = []
                for i in range(len(tokens)):
                    if 0 == i:
                        print >> stock_predict_out, "%s," % tokens[i],
                    elif 1 == i:
                        print >> stock_predict_out, "%s,1," % tokens[i],
                    else:
                        l_features.append(float(tokens[i].strip()))
                l_features2 = []
                l_features2.append(l_features)
                np_features = np.array(l_features2)
                pred = model_predictor.predict_proba(np_features)
                print >> stock_predict_out, "%f" % pred[0,1]
            stock_predict_out.close()
        shutil.move("../data/model_tunner.list","../data/model_tunner.list.bk")

        time.sleep(1)

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
