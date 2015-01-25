#!/usr/bin/env python
#@author redbin@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
# parse the command paramaters
from optparse import OptionParser

from model_base import *
import logging

import subprocess


logdir = local_path + '/../log/'
if not os.path.exists(logdir):
    os.makedirs(logdir)
logging.basicConfig(level = logging.DEBUG,
    format = '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    filename = logdir + '%s.log' % (os.path.basename(sys.argv[0]),), 
    filemode = 'a')
spxdata = get_stock_date2data('SPX')

class ExtractorBase: # {{{
    def __init__(self, symbol, stock_data, window):
        self.symbol = symbol
        self.stock_data = stock_data
        self.window = window
        self.spx = spxdata
    def extract_features_and_classes(self):
        assert(False)
    def extract_last_features(self):
        assert(False)
# }}}

class Extractor4(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.stock_data)-self.window-1):
            for  j in range(self.window):
                inc = self.stock_data[i+j+1]["open"] * 1.0 / self.stock_data[i+j]["close"]
                inc = inc / (self.spx[self.stock_data[i+j+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i+j]["date"]]["close"])
                
                ret +=  str(inc) + ","
                inc = self.stock_data[i+j+1]["high"] * 1.0 / self.stock_data[i+j]["close"]
                inc = inc / (self.spx[self.stock_data[i+j+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i+j]["date"]]["close"])
                
                ret +=  str(inc) + ","
                inc = self.stock_data[i+j+1]["low"] * 1.0 / self.stock_data[i+j]["close"]
                inc = inc / (self.spx[self.stock_data[i+j+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i+j]["date"]]["close"])
                
                ret +=  str(inc) + ","
                inc = self.stock_data[i+j+1]["close"] * 1.0 / self.stock_data[i+j]["close"]
                inc = inc / (self.spx[self.stock_data[i+j+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i+j]["date"]]["close"])
                ret +=  str(inc) + ","
            classes = 0
            nc =  self.stock_data[i+self.window + 1]["close"] / self.stock_data[i+self.window]["close"] 
            nc =  nc / (self.spx[self.stock_data[i+self.window + 1]["date"]]["close"]*1.0 / self.spx[self.stock_data[i+self.window]["date"]]["close"]) 
            if nc > 1.0:
                 classes = 1
            ret += "%d,%s,%f,%f,%s" % (classes,
                                       self.symbol,
                                       self.stock_data[i+self.window]["close"],
                                       self.stock_data[i+self.window+1]["close"],
                                       self.stock_data[i+self.window]["date"]) + "\n"
        return ret
    # }}}

    def extract_last_features(self): #{{{
        ret = ""
        ret += self.symbol + ","
        ret += str(self.stock_data[-1]["date"]) + ","
        for i in range(len(self.stock_data)-self.window-1, len(self.stock_data)-1):
            inc = self.stock_data[i+1]["open"]*1.0 / self.stock_data[i]["close"]
            inc = inc / (self.spx[self.stock_data[i+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i]["date"]]["close"])
            ret += str(inc) + "," 
            inc = self.stock_data[i+1]["high"]*1.0 / self.stock_data[i]["close"]
            inc = inc / (self.spx[self.stock_data[i+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i]["date"]]["close"])
            ret += str(inc) + "," 
            inc = self.stock_data[i+1]["low"]*1.0  / self.stock_data[i]["close"]
            inc = inc / (self.spx[self.stock_data[i+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i]["date"]]["close"])
            ret += str(inc) + "," 
            inc = self.stock_data[i+1]["close"]*1.0/ self.stock_data[i]["close"]
            inc = inc / (self.spx[self.stock_data[i+1]["date"]]["close"]*1.0/self.spx[self.stock_data[i]["date"]]["close"])
            if i != (len(self.stock_data)-2):
                ret += str(inc) + "," 
            else:
                ret += str(inc) + "\n" 
        return ret
    # }}}

def main(options, args): # {{{
    options.window = int(options.window)

    if options.utildate == None:
        print "the utildate is NULL"
        sys.exit(1)
    yhoo = get_stock_data_one_day("A", options.utildate)
    if not "close" in yhoo:
        print "the utildate(%s) has no data" % options.utildate
        sys.exit(1)
    d_train     = options.output + "/" + options.utildate ;
    if not os.path.exists(d_train) : os.makedirs(d_train)
    f_train     = open(d_train +  "/train.csv", "w")
    f_last      = open( d_train + "/last.csv" , "w")

    # get the extractor
    Extractor = globals()[options.extractor]
    file_list = get_file_list(options.stocks_path)
    file_list.sort()
    stock_num = 0
    file_list_len = len(file_list)
    train_len = file_list_len * 0.8
    index = 0
    for f in file_list[0:file_list_len]:
        index +=1
        stock_num += 1
        if stock_num % 10 == 0:
            logging.debug("build the %d's stock" % stock_num)
        symbol = get_stock_from_path(f)
        stock_data = get_stock_data2(f, 30 * 2 + options.window, options.utildate)
        if len(stock_data) == 0:
            continue
        if stock_data[-1]["date"] != options.utildate:
            logging.error("symbol(%s) date(%s) not equals to %s", symbol, stock_data[-1]["date"], options.utildate)
        extractor = Extractor(symbol, stock_data, options.window)
        if len(stock_data)  < options.limit:
            logging.debug("%s is too short(%d)!" % (symbol, len(stock_data)))
            continue
        foutput = f_train
        print >> foutput, "%s" %  \
            extractor.extract_features_and_classes(),
        print >> f_last, "%s" % \
                extractor.extract_last_features(),
    f_train.close()
    f_last.close()
    #cmds = """cat  """+d_train + "/train.csv" + """  | awk -F"," '{print $(NF-3)"\1\2"$0}' | sort -k1 | awk -F"\1\2" '{print $2}' > """ + d_train + "/tmp1" 
    cmds = """cat  """+d_train + "/train.csv" + """  | awk -F"," '{print $(NF)"\1\2"$0}' | sort -k1 | awk -F"\1\2" '{print $2}' > """ + d_train + "/tmp1" 
    print cmds
    os.system(cmds)
    #cmds = """cat  """+d_train + "/train.csv" + """  | awk -F"," '{print $(NF-3)"\1\2"$0}' | sort -k1 | awk -F"\1\2" '{print $2}' | awk -F"," '{print $(NF-3)}'> """ + d_train + "/tmp2" 
    cmds = """cat  """+d_train + "/train.csv" + """  | awk -F"," '{print $(NF)"\1\2"$0}' | sort -k1 | awk -F"\1\2" '{print $2}' | awk -F"," '{print $(NF)}'> """ + d_train + "/tmp2" 
    print cmds
    os.system(cmds)
    lines = open(d_train + "/tmp1").readlines()
    dates = open(d_train + "/tmp2").readlines()
    assert(len(lines) == len(dates))
    len1 = len(lines) * 0.95
    f_test_train = open(d_train + "/test_train.csv", "w")
    f_test_test = open(d_train + "/test_test.csv" , "w")
    for i in range(len(lines)):
        if i < len1:
            print >> f_test_train, lines[i].strip()
        else:
            break
    j = i 
    while dates[j] == dates[i] and j < len(lines):
        print >> f_test_train, lines[j].strip()
        j+=1
    for i in range(j, len(lines)):
        print >> f_test_test, lines[i].strip()
    f_test_train.close()
    f_test_test.close()



# }}}

def parse_options(parser): #{{{
    """
    parser command line
    """
    parser.add_option("--extractor", dest="extractor",action = "store", \
            default="Extractor5", help = "the extractor to use")
    parser.add_option("--window", type="int", dest="window",action = "store", \
            default=60, help = "the history price window")
    parser.add_option("--output", dest="output",action = "store", \
            default=None, help = "the output directory")
    parser.add_option("--stocks_path", dest="stocks_path",action = "store", \
            default="/home/work/workplace/stock_data/", \
            help = "the stocks data directory")
    parser.add_option("--limit", type="int", dest="limit",action = "store", \
            default=30, \
            help = "the limit length of stock data")
    parser.add_option("--utildate", dest="utildate",action = "store", default=None, help = "the last date to train")
    return parser.parse_args()
# }}}

if __name__ == "__main__": #{{{
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
# }}}
