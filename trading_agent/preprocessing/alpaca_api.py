import requests
import json as json
from datetime import datetime
import alpaca_trade_api as t
import logging
import pandas as pd
import pytz
import time

endpoint = "https://data.alpaca.markets/v1"

#headers = json.loads(open("account.json", 'r').read())

headers = {
    "APCA-API-KEY-ID": "PKORD0QGJF77SV37G8SJ",
    "APCA-API-SECRET-KEY": "CA1WMXF5e6uojeEb6BtDLjMy9RT0NaLA4g3beU69"
}


def hist_data(symbol, dataframe, timeframe="day", limit=1000, start="", end="", after="", until=""):
    '''Returns the historical bar data for a group of stocks '''
    df_data = {}
    # Get Requests for Bar Data
    bar_url = endpoint + "/bars/{}".format("day")

    params = {
        "symbols": symbol,
        "start": start,
        "end": end,
        "limit": limit,
        "timeframe": "day"
    }

    r = requests.get(bar_url, headers=headers, params=params)

    json_dump = r.json()
    # loop through stock data
    for symbol in json_dump:
        # convert json into pandas dataframe
        temp = pd.DataFrame(json_dump[symbol])
        temp.rename({"t":"datadate", "o": "open", 'h':'high', 'l':'low', "c":"close", 'v':'volume'}, axis=1, inplace=True)

        temp['tic'] = symbol

        dataframe = pd.concat([dataframe, temp])

    return dataframe


def getData(tickers, start, end, limit):
    dataframes = dict()
    for symbol in tickers:
        dataframes[symbol] = pd.DataFrame()#columns = [symbol])

    # time is in seconds

    # starttime = time.time()
    # # will go for 8 hours
    # # timeout = starttime + 60 * 5 # 8 hrs 60 * 8
    # timeout = starttime + 1

    # while time.time() <= timeout:
    # print("****************************************************")
    for company in tickers:
        # print("printing data for {} at {}".format(company, time.time()))
        dataframes[company] = hist_data(company, dataframes[company], limit=limit, start=start, end=end)

    final_prices = dataframes[tickers[0]]

    for key, value in dataframes.items():
        final_prices = pd.concat([final_prices, value])

    final_prices = final_prices.sort_values(by=['datadate'])

    for index, row in final_prices.iterrows():
        year = str(datetime.fromtimestamp(int(row['datadate'])).year)
        month = str(datetime.fromtimestamp(int(row['datadate'])).month)
        if len(month) == 1:
            month = "0" + month
        day = str(datetime.fromtimestamp(int(row['datadate'])).day)
        if len(day) == 1:
            day = "0" + day
        # print(year+month+day)
        # row['datadate'] = year+month+day
        final_prices.at[index, 'datadate'] = year+month+day

    final_prices = final_prices.reset_index()
    final_prices = final_prices.drop(["index"], axis=1)
    final_prices = final_prices.drop_duplicates()
    return final_prices

# print(getData(["GOOG", "MSFT", "AAPL", "AMZN", "FB"]))
