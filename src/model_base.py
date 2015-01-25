#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author 
import sys,os
import datetime
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

def main():
    pass


import talib
import numpy

class feature_builder_ohc(): #{{{
    def __init__(self, feature_func, builder_list):
        self.feature_build_func = feature_func
        builder_list.append(self)

    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = self.feature_build_func(high_price, low_price, close_price)
        return result

    def name(self):
        return self.feature_build_func.__name__
#}}}

def get_file_list(rootdir): #{{{
    """hongbin0908@126.com
    a help function to load test data.
    """
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(rootdir, f))
         
    return file_list
# }}}

def get_stock_from_path(pathname): #{{{
    """
    from /home/work/workplace/pytrade/strategy_mining/utest_data/stocks/AAPL.csv to AAPL
    """
    return os.path.splitext(pathname.split("/")[-1])[0]
# }}}

def load_data(filename, dates, open_price, high_price, low_price, close_price, adjust_price, volume_list): #{{{
    fd = open(filename, "r")
    for j in fd:
        try:
            line_list = j.rstrip().split(",")
            date_str = line_list[0]
            open_p = float(line_list[1])
            volume = float(line_list[5])
            high_p = float(line_list[2])
            low_p = float(line_list[3])
            close_p = float(line_list[4])
            dates.append(date_str)
            open_price.append(open_p)
            high_price.append(high_p)
            low_price.append(low_p)
            close_price.append(close_p)
            volume = float(line_list[5])
            adjust_p = float(line_list[6])
            adjust_price.append(adjust_p)
            volume_list.append(volume)
        except Exception, e:
            continue
    fd.close()
    return 0
# }}}

def get_stock_data(filename, str_startdate, str_utildate): #{{{
    """
    input filename : the path of stock data
    """
    dt_utildate = parse_date_str(str_utildate)
    dt_startdate = parse_date_str(str_startdate)
    dates = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    adjust_close_prices = []
    volumes = []
    load_data(filename, dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes)
    dates.reverse()
    open_prices.reverse()
    high_prices.reverse()
    low_prices.reverse()
    close_prices.reverse()
    adjust_close_prices.reverse()
    volumes.reverse()
    length = len(dates)
    dates2 = [] 
    open_prices2 = []
    high_prices2 = []
    low_prices2 = []
    close_prices2 = []
    adjust_close_prices2 = []
    volumes2 = []
    rs = []
    for i in range(-1 * length, 0):
        dt_cur_date = parse_date_str(dates[i])
        
        if dt_cur_date <= dt_utildate and dt_cur_date >= dt_startdate:
            d_cur = {}
            #[{"date":xx,"open":xx,"high":xx,"low":xx,"close":xx,"adj_close":xx,"volume":xx},...,{...}]
            d_cur["date"] = dates[i] 
            d_cur["open"] = open_prices[i]
            d_cur["high"] = high_prices[i]
            d_cur["low"] = low_prices[i]
            d_cur["close"] = close_prices[i]
            d_cur["adj_close"] = adjust_close_prices[i]
            d_cur["volume"] = volumes[i]
            rs.append(d_cur)
    return rs
#}}}

def get_stock_data2(filename, length, str_utildate): # {{{
    delta = datetime.timedelta(days=length)
    dt_utildate = parse_date_str(str_utildate)
    dt_startdate = dt_utildate - delta
    str_startdate = dt_startdate.strftime('%Y-%m-%d')
    return get_stock_data(filename, str_startdate, str_utildate)
# }}}

def get_stock_data_one_day(sym, datestr,stock_root="/home/work/workplace/stock_data/"): #{{{
    """
    描述: 获取指定一天的股票数据
    输入: 股票代码 时间(YYYY-MM-DD)
    输出: dict {"open":xx,"high":xx,"low":xx,"close":xx,"adj_close":xx,"volume":xx}

    """
    if len(datestr) != 10:
        assert(False)
    res = {}
    filename = os.path.join(stock_root, sym + ".csv")
    if not os.path.exists(filename): 
        print "filename not exists : " , filename
        assert(False)
    lines = open(filename, "r").readlines() ; assert(len(lines)>1)
    for line in lines[1:]:
        terms = line.rstrip().split(",")
        strdate = terms[0]
        if  len(strdate) != 10:
            print "the date string format error[date:%s][line:%s]" % (strdate, line)
            assert(False)
        if strdate != datestr :
            continue
        res["open"]  = float(terms[1])
        res["high"]  = float(terms[2])
        res["low"]   = float(terms[3])
        res["close"] = float(terms[4])
        res["volume"]  = float(terms[5])
        res["adj_close"] = float(terms[6])

        # random check
        assert( res["low"] <= res["high"]  )
        assert( res["low"] <= res["open"]  )
        assert( res["low"] <= res["close"]  )
        
        break
    return res;
# }}}

def get_stock_data_span_day(sym, datestr,span, stock_root="/home/work/workplace/stock_data/"): #{{{
    """
    描述: 获取指定一天后span天的股票数据
    输入: 股票代码 时间(YYYY-MM-DD)
    输出: dict {"open":xx,"high":xx,"low":xx,"close":xx,"adj_close":xx,"volume":xx}

    """
    if len(datestr) != 10:
        assert(False)
    res = {}
    filename = os.path.join(stock_root, sym + ".csv")
    if not os.path.exists(filename): 
        print "filename not exists : " , filename
        assert(False)
    lines = open(filename, "r").readlines()[1:] ; assert(len(lines)>1)
    index = 0
    for index in range(1, len(lines)):
        line = lines[index]
        terms = line.rstrip().split(",")
        strdate = terms[0]
        if  len(strdate) != 10:
            print "the date string format error[date:%s][line:%s]" % (strdate, line)
            assert(False)
        if strdate != datestr :
            continue
        line = lines[index - span]
        terms = line.rstrip().split(",")
        res["open"]  = float(terms[1])
        res["high"]  = float(terms[2])
        res["low"]   = float(terms[3])
        res["close"] = float(terms[4])
        res["volume"]  = float(terms[5])
        res["adj_close"] = float(terms[6])
        res["date"] = terms[0]

        # random check
        assert( res["low"] <= res["high"]  )
        assert( res["low"] <= res["open"]  )
        assert( res["low"] <= res["close"]  )
        
        break
    return res;
# }}}

def get_stock_date2data(symbol): #{{{
    """
    input filename : the path of stock daily data
    """
    rs = {}
    dates = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    adjust_close_prices = []
    volumes = []
    filename = os.path.join("/home/work/workplace/stock_data/", symbol + ".csv")
    load_data(filename, dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes)
    dates.reverse()
    open_prices.reverse()
    high_prices.reverse()
    low_prices.reverse()
    close_prices.reverse()
    adjust_close_prices.reverse()
    volumes.reverse()
    length = len(dates)
    dates2 = [] 
    open_prices2 = []
    high_prices2 = []
    low_prices2 = []
    close_prices2 = []
    adjust_close_prices2 = []
    volumes2 = []
    for i in range(-1 * length, 0):
        dt_cur_date = parse_date_str(dates[i])
        rs[dates[i]] = {}
        rs[dates[i]]["open"] = open_prices[i]
        rs[dates[i]]["high"] = high_prices[i]
        rs[dates[i]]["low"] = low_prices[i]
        rs[dates[i]]["close"] = close_prices[i]
        rs[dates[i]]["adjust"] = adjust_close_prices[i]
        rs[dates[i]]["volumes"] = volumes[i]
        rs[dates[i]]["date"] = dates[i] 
    return rs
#}}}
def get_date_str(): # {{{
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d')
# }}}

def parse_date_str(date_str): # {{{
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')
# }}}

def isTraderDay(datestr): #{{{
    """
    判断datestr是否为交易日
    """
    yhoo = get_stock_data_one_day("A", options.utildate)
    if not "close" in yhoo:
        return False
    return True
# }}}

if __name__ == '__main__':
    main()
