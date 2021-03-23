# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
from preprocessing.alpaca_api import *
from preprocessing.GetStocks import *
# config
from config.config import *
# model
from model.models import *
import os
from stable_baselines import A2C

from run_DRL import run_model
import alpaca_trade_api as alpaca

headers = json.loads(open("account.json", 'r').read())
# api = alpaca.REST(headers)
api = alpaca.REST(
    headers['APCA-API-KEY-ID'],
    headers['APCA-API-SECRET-KEY'],
    'https://paper-api.alpaca.markets', api_version='v2'
)

account = api.get_account()

HMAX_NORMALIZE = 100
STOCK_DIM = 20


def load_model(tickers):
    '''Load in the pretrained model from the trained models folder '''
    # model = run_model(tickers,start="2020-01-01T09:30:00-04:00", end="2020-12-31T09:30:00-04:00")
    model = A2C.load("trained_models/2021-03-22 18:25:09.528982/A2C_30k_dow_120.zip")

    return model


def makeTrades(df, model):
    '''predicts on current state using pretrained model'''
    mappings = dict()
    i = 0

    for index, row in df.iterrows():
        mappings[i] = row['tic']
        i += 1

    print(mappings)

    data = [account.buying_power] + \
                  df.adjcp.values.tolist() + \
                  [0]*STOCK_DIM + \
                  df.macd.values.tolist() + \
                  df.rsi.values.tolist() + \
                  df.cci.values.tolist() + \
                  df.adx.values.tolist()


    actions, _states = model.predict(data)

    actions = actions * HMAX_NORMALIZE
    # print(actions)
    # temp_score = sum(actions)
    # temp_score = [actions[i]/temp_score for i in range(len(actions))]#actions/temp_score
    # print(temp_score)
    # # actions = actions * temp_score * HMAX_NORMALIZE
    # # print(actions)
    # # # qty = account.buying_power * (abs(int(actions[index]))/temp_score)
    # # print((actions/temp_score))
    # print(type(temp_score))
    # qty = float(account.buying_power*temp_score[0])
    # print(qty)

    argsort_actions = np.argsort(actions)

    sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
    buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

    portfolio = api.list_positions()

    for index in sell_index:
        print('take sell action {}'.format(mappings[index]))
        # api.submit_order(symbol=mappings[index],qty=abs(int(actions[index])),side='sell',type='market',time_in_force='day')
        # api.submit_order(symbol=mappings[index],notional=20000,side='sell',type='market',time_in_force='day')
        # api.submit_order(symbol=mappings[index],qty=abs(int(account.buying_power*temp_score[index])),side='sell',type='market',time_in_force='day')
        # pass

    for index in buy_index:
        print('take buy action: {}'.format(mappings[index]))
        # api.submit_order(symbol=mappings[index],qty=int(actions[index]),side='buy',type='market',time_in_force='day')
        # api.submit_order(symbol=mappings[index],notional=20000,side='buy',type='market',time_in_force='day')
        # api.submit_order(symbol=mappings[index],qty=abs(int(account.buying_power*temp_score[index])),side='sell',type='market',time_in_force='day')



if __name__ == "__main__":
    # tickers = get_highest_movers()
    tickers = ['DISCA', 'ENPH', 'PENN', 'VIAC', 'HFC', 'WAT', 'NVR', 'UAL', 'DISCK', 'CF', 'BWA', 'APA', 'MRO', 'FANG', 'TFX', 'OXY', 'ROST', 'CCL', 'ALK', 'LUMN']
    print(tickers)

    model = load_model(tickers)

    # isOpen = self.alpaca.get_clock().is_open
    # while(not isOpen):
    #     clock = self.alpaca.get_clock()
    #     openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
    #     currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
    #     timeToOpen = int((openingTime - currTime) / 60)
    #     print(str(timeToOpen) + " minutes til market open.")
    #     time.sleep(60)
    #     isOpen = self.alpaca.get_clock().is_open

    # Get previous day stock information from alpaca as df
    data = preprocess_data(tickers, limit=2)
    data = data[(data.datadate >= data.datadate.max())]
    data = data.reset_index()
    data = data.drop(["index"], axis=1)
    data = data.fillna(method='ffill')
    print(data)

    makeTrades(data, model)

    1,000,043.38

    #3,803,810.34 Buying Power

    #3,803,810.34
