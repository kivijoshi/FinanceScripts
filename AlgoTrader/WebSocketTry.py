# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:58:57 2017

@author: kaust
"""

from kiteconnect import WebSocket

# Initialise.
kws = WebSocket("syehwjh8tmbedsv5", "c612bf39b1f2a1bac5621b1402cb50d5", "ZV4007")

# Callback for tick reception.
def on_tick(tick, ws):
    print tick

# Callback for successful connection.
def on_connect(ws):
    # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
    ws.subscribe([738561, 5633])

    # Set RELIANCE to tick in `full` mode.
    ws.set_mode(ws.MODE_FULL, [738561])

# Assign the callbacks.
kws.on_tick = on_tick
kws.on_connect = on_connect

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
kws.connect()