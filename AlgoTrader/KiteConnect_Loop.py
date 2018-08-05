# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:04:01 2017

@author: kaust
"""

import time
from kiteconnect import KiteConnect

'''
{'access_token': '97vj69chbvqptjt4kxnkykq36v21oq73',
 'broker': 'ZERODHA',
 'email': 'KAUSTUBH.KIVIJOSHI@GMAIL.COM',
 'exchange': ['BSE', 'BFO', 'MCX', 'CDS', 'NSE', 'NFO'],
 'login_time': '2017-01-11 19:01:40',
 'member_id': 'ZERODHA',
 'order_type': ['MARKET', 'LIMIT', 'SL', 'SL-M'],
 'password_reset': False,
 'product': ['BO', 'CO', 'CNC', 'MIS', 'NRML'],
 'public_token': '3773197349ace7ee00ec124567402971',
 'user_id': 'ZV4007',
 'user_name': 'VIJAY  LAXMAN JOSHI',
 'user_type': 'investor'}
 '''

kite = KiteConnect(api_key="syehwjh8tmbedsv5")

kite.set_access_token("yya4flub8jbmuvwup7r6qg8em77dekls")

Market_Open_Time = 33300
Market_Close_Time = 55800

starttime=time.time()
cnt = 0
quote = kite.quote("NSE","MCX")
print(quote)
with open("MCX_monthlyData_reduced.txt", "a") as outF:
          outF.write("\n")
          date = str(quote['last_time'])
          close = str(quote['last_price'])
          line = date + ',' + close
          outF.write(line)
outF.close()
      
#while True:
#  cnt = cnt + 1
#  print(time.time())
#  quote = kite.quote("NSE","MCX")
#  print(quote)
#  print(cnt)
#  time.sleep(60.0 - ((time.time() - starttime) % 60.0))
#  timedate = quote['last_time']
#  seconds = int(timedate[-2:])
#  minutes = int(timedate[-5:-3])
#  hours = int(timedate[-8:-6])
#  totalseconds = (hours*60*60) + (minutes*60) + seconds
#  if(totalseconds > Market_Open_Time and totalseconds < Market_Close_Time):
#      with open("MCX_monthlyData.txt", "a") as outF:
#          outF.write("appended text")
#          outF.write("\n")
#          close = str(quote['ohlc']['close'])
#          date = str(quote['ohlc']['last_time'])
#          high = str(quote['ohlc']['high'])
#          low = str(quote['ohlc']['low'])
#          opn = str(quote['ohlc']['open'])
#          volume = str(quote['volume'])
#          line = close + ',' + date + ',' + high + ',' + low + ',' + opn + ',' + volume
#          outF.write(line)
#      outF.close()

      