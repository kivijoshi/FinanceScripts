# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
from kiteconnect import KiteConnect
https://kite.trade/connect/login?api_key=syehwjh8tmbedsv5'
kite.request_access_token("le835wm6duvkbnonxbon4yqmrakvic6o", secret="2jqxarkcq0vu7avd918syg5fkejriti7")
'''
from kiteconnect import KiteConnect
import json

'''
{'access_token': 'yya4flub8jbmuvwup7r6qg8em77dekls',
 'broker': 'ZERODHA',
 'email': 'KAUSTUBH.KIVIJOSHI@GMAIL.COM',
 'exchange': ['BSE', 'BFO', 'MCX', 'CDS', 'NSE', 'NFO'],
 'login_time': '2017-01-13 23:47:38',
 'member_id': 'ZERODHA',
 'order_type': ['MARKET', 'LIMIT', 'SL', 'SL-M'],
 'password_reset': False,
 'product': ['BO', 'CO', 'CNC', 'MIS', 'NRML'],
 'public_token': '627cce464b577654df70d1b84e536b8c',
 'user_id': 'ZV4007',
 'user_name': 'VIJAY  LAXMAN JOSHI',
 'user_type': 'investor'}
 '''
kite = KiteConnect(api_key="syehwjh8tmbedsv5")

kite.set_access_token("yya4flub8jbmuvwup7r6qg8em77dekls")

#print(kite.quote("NSE","ACC"))
A = kite.historical(7982337,"2016-12-14","2017-01-13","minute")

outF = open("MCX_monthlyData_reduced.txt", "w")
line = 'DATE' + ',' + 'OPEN'
outF.write(line)
for data in A:
    outF.write("\n")
    close = str(data['close'])
    date = str(data['date'])
    #high = str(data['high'])
    #low = str(data['low'])
    #opn = str(data['open'])
    #volume = str(data['volume'])
    line = date + ',' + close
    outF.write(line)
outF.close
#A = kite.historical(7982337,"2016-10-13","2017-01-11","5minute")
#outF = open("MCX_90D_5Min_Data.txt", "w")
#line = 'close' + ',' + 'date' + ',' + 'high' + ',' + 'low' + ',' + 'open' + ',' + 'volume'
#outF.write(line)
#for data in A:
#    outF.write("\n")
#    close = str(data['close'])
#    date = str(data['date'])
#    high = str(data['high'])
#    low = str(data['low'])
#    opn = str(data['open'])
#    volume = str(data['volume'])
#    line = close + ',' + date + ',' + high + ',' + low + ',' + opn + ',' + volume
#    outF.write(line)

#A = kite.historical(7982337,"2016-07-15","2017-01-11","15minute")
#outF = open("MCX_2016-07-15_2017-01-11_15min_Data.txt", "w")
#line = 'close' + ',' + 'date' + ',' + 'high' + ',' + 'low' + ',' + 'open' + ',' + 'volume'
#outF.write(line)
#for data in A:
#    outF.write("\n")
#    close = str(data['close'])
#    date = str(data['date'])
#    high = str(data['high'])
#    low = str(data['low'])
#    opn = str(data['open'])
#    volume = str(data['volume'])
#    line = close + ',' + date + ',' + high + ',' + low + ',' + opn + ',' + volume
#    outF.write(line)

#A = kite.historical(7982337,"2016-01-11","2017-01-11","60minute")
#outF = open("MCX_2016-01-11_2017-01-11_60min_Data.txt", "w")
#line = 'close' + ',' + 'date' + ',' + 'high' + ',' + 'low' + ',' + 'open' + ',' + 'volume'
#outF.write(line)
#for data in A:
#    outF.write("\n")
#    close = str(data['close'])
#    date = str(data['date'])
#    high = str(data['high'])
#    low = str(data['low'])
#    opn = str(data['open'])
#    volume = str(data['volume'])
#    line = close + ',' + date + ',' + high + ',' + low + ',' + opn + ',' + volume
#    outF.write(line)

#A = kite.historical(7982337,"2016-10-13","2017-01-11","3minute")
#outF = open("MCX_2016-10-13_2017-01-11_3min_Data.txt", "w")
#line = 'close' + ',' + 'date' + ',' + 'high' + ',' + 'low' + ',' + 'open' + ',' + 'volume'
#outF.write(line)
#for data in A:
#    outF.write("\n")
#    close = str(data['close'])
#    date = str(data['date'])
#    high = str(data['high'])
#    low = str(data['low'])
#    opn = str(data['open'])
#    volume = str(data['volume'])
#    line = close + ',' + date + ',' + high + ',' + low + ',' + opn + ',' + volume
#    outF.write(line)
