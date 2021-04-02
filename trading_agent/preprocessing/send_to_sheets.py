import requests
import json as json
from datetime import datetime
import alpaca_trade_api as t
import logging
import pytz
import time
import csv

# google sheets api creds
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build 
from google.auth.transport.requests import Request
from googleapiclient import discovery

from datetime import date 

import os 
import json 

# input : year(date), num of stocks
# output: list of volatile stocks

import yfinance as yf
# from pandas_datareader import data as pdr
import pandas as pd
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# Google Sheets Authentication 
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]


# then the following should work

# scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
# creds = ServiceAccountCredentials.from_json_keyfile_dict(create_keyfile_dict(), scope)
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
service = discovery.build('sheets', 'v4', credentials=creds)

# creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
# creds = ServiceAccountCredentials.from_json_keyfile_name(cred_dict, scope)
client = gspread.authorize(creds)



def get_stock_symbols():
  sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  df = sp500[0]
  symbols = df['Symbol'].tolist()
  return symbols


def df_lookup(df, key_row, key_col):
  try:
    return df.iloc[key_row][key_col]
  except IndexError:
    return 0

def get_movement_list(stocks, period):
  movement_list = []
  f = open("stock_changes.csv", "w+")
  stock_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  stock_writer.writerow(["stock", "delta_percent", "delta_price"]) # add header to csv
  for stock in stocks:
    # get history
    curr_stock = yf.Ticker(stock)
    # print(curr_stock.info())
    
    hist = curr_stock.history(period = period) #lookback 1 day

    low = float(10000)
    high = float(0)
    # print(curr_stock.info)
    for day in hist.itertuples(index=True, name='Pandas'):
      if day.Low < low:
        low = day.Low
      if high < day.High:
        high = day.High
    #for zero division error handling
    # if low == 0:
    #   delta_percent = 0
    # else:
    delta_percent = 100 * (high - low) / low #check for division by 0
    Open = df_lookup(hist, 0, "Open")

    # some error handling:
    if len(hist >= 5):
      Close = df_lookup(hist, 4, "Close")
    else :
      Close = Open

    if (Open == 0):
      delta_price = 0
    else:
      delta_price = 100 * (Close - Open) / Open

    # print(stock+" "+str(delta_percent)+ " "+ str(delta_price))
    pair = [stock, delta_percent, delta_price]
    movement_list.append(pair)
    stock_writer.writerow(pair)
  #close the txt file
  f.close()
  return curr_stock


def send_to_spreadsheet(stocks):
    '''sends data to google sheet '''
        
    spreadsheet_id = ''

    # range_ = 'Sheet2!A1:B1'
    # sheet2 = client.open("customers_texted").worksheet('Sheet2')
    value_input_option = "USER_ENTERED"
    # row = sheet2.row_values()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    client = gspread.authorize(creds)
    sheet = client.open('Daily_Volatile_Stocks').worksheet('Sheet1')

    data = sheet.get_all_records()

    rows_filled = len(data) + 1 

    counter = rows_filled + 1
    today = date.today()
    
    for stock in stocks:

        ticker = [[str(today), stock]]
        
        range_ = f'Sheet1!A{counter}:B{counter}'
        
        counter += 1 
        
        result = service.spreadsheets().values().update(spreadsheetId=spreadsheet_id, range=range_, 
                                                valueInputOption=value_input_option, body={'values':ticker})
        response = result.execute()
        # print('{0} cells updated.'.format(response.get('updatedCells')))
        


# time get movement list function #started 6:36, end 6:40
# stocks = get_stock_symbols()
# start = time.perf_counter()
# get_movement_list(stocks, "1d")
# end = time.perf_counter()
# print(end - start)

def get_highest_movers():
    stocks = get_stock_symbols()
    get_movement_list(stocks, "1d")

    #read the stock_changes csv file
    stocks = pd.read_csv('stock_changes.csv')
    #sort by delta percent
    sorted_stocks = stocks.sort_values('delta_percent', ascending=False)
    # #take the top 20 values
    most_volatile_stocks = sorted_stocks.head(20)
    # sends data to spreadsheet
    send_to_spreadsheet((most_volatile_stocks['stock'].tolist()))


    return 'Successfully updated spreadsheet with the most volatile stocks'


get_highest_movers()