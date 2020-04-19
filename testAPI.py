# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:20:46 2020

@author: ArturoA
"""

from flask import Flask, jsonify, request, json

# local url
url = 'http://127.0.0.1:5000/' # change to your url
data = {'0': 0
      , '1': 0.6
      , '2': 0.506944
      , '3': 0.6
      , '4': 0.055236
      , '5': 0.374126
      , '6': 0.724907
      , '7': 0.199052
      , '8': 0.786765
      , '9': 0.302954
      , '10': 0.261838
      , '11': 0.031014
      , '12': 0.35467
      , '13': 0.223484
      , '14': 0.405913
      , '15': 1
      , '16': 0.034749}
data = json.dumps(data)
sent_request = request.post(url, data)
print(send_request)
print(send_request.json())