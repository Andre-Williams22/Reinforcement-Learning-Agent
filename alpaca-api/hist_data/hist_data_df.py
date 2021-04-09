import requests
import json as json
from datetime import datetime
import alpaca_trade_api as t
import logging 
import pandas as pd 
import pytz 
import time 


endpoint = "https://data.alpaca.markets/v1"

headers = json.loads(open("account.json", 'r').read())


def hist_data(symbols, timeframe="15Min", limit=1, start="", end="", after="", until=""):
    '''Returns the historical bar data for a group of stocks '''
    df_data = {}
    # Get Requests for Bar Data
    bar_url = endpoint + "/bars/{}".format(timeframe)

    params = {
        "symbols": symbols,
        "limit": limit,
        "start": start,
        "end": end,
        "after": after,
        "until": until
    }

    r = requests.get(bar_url, headers=headers, params=params)

    json_dump = r.json()
    # loop through stock data 
    for symbol in json_dump:
        # print("symbol = ",symbol)
        # print(json_dump[symbol])
        # convert json into pandas dataframe 
        temp = pd.DataFrame(json_dump[symbol])
        temp = temp[['t','c']]
        temp.rename({"t":"time", "c":"close"}, axis=1, inplace=True)
        temp["time"] = pd.to_datetime(temp["time"], unit="s")
        temp.set_index("time", inplace=True)
        temp.index = temp.index.tz_localize("UTC").tz_convert("America/Indiana/Petersburg")
        temp.between_time('09:31', '16:00')
        
        # append data to df data 
        df_data[symbol] = temp
        
    #df_data = pd.DataFrame(df_data)
    
        
    return df_data

#data_dump = hist_data("FB,AMZN,INTC,MSFT,AAPL,GOOG,CSCO,CMCSA,ADBE,NVDA,NFLX,PYPL,AMGN,AVGO,TXN,CHTR,QCOM,GILD,FISV,BKNG,INTU,ADP,CME,TMUS,MU", timeframe="5Min", limit=200) #"NIO,AAPL,PLTR"

data_dump = hist_data('MSFT')
max_pos = 1000


# print(int(max_pos/hist_data('AAPL')['AAPL']["close"].iloc[-1]))
print(data_dump)