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
import ssl
import json 


account = api.get_account()
HMAX_NORMALIZE = 100
STOCK_DIM = 20

def load_model(tickers):
    '''Load in the pretrained model from the trained models folder '''
    # model = run_model(tickers,start="2020-01-01T09:30:00-04:00", end="2020-12-31T09:30:00-04:00")
    model = A2C.load("trained_models/2021-03-22 18:25:09.528982/A2C_30k_dow_120.zip")

    return model

def buy_stock(ticker, num_of_shares):
    '''Buys stock if we have the funds available'''

    cash_in_hand = float(account.buying_power)
 
    # grab current price of stock
    current_price = api.get_barset(ticker, 'day', limit=1)
    price = current_price[ticker]
    cprice = price[-1].c
    cost = round((num_of_shares*cprice)-0.5)
    
    # determine if we can afford stock 
    if cash_in_hand > cost:
        shares = round((int(num_of_shares)))
        
        
        try: 
            # alpaca api request buy order
            api.submit_order(symbol=ticker, qty=shares, side='buy', type='market', time_in_force='day')
            # print('cash', float(account.buying_power))
            # print('cost', cost)
        except:
            return "Need more cash, can't buy anything"
            

def sell_stock(ticker, num_of_shares):
    ''' sells stock in the market using shares we have and shares we don't have via shorting '''
    
    positions = api.list_positions()
    if num_of_shares > 0:# can't short sell stocks
        print(ticker, round(int(num_of_shares)))
    # submit alpaca request buy order   could be shorting which is betting on market going down
        
        for position in positions:
            if position.symbol == ticker and abs(int(num_of_shares)) <= int(position.qty):
        ## regular sell order 
                api.submit_order(symbol=ticker,qty=round(int(num_of_shares)),side='sell',type='market',time_in_force='day')
    else:
        try: 
            print('shorting stock: ', num_of_shares)
            #Short the stock 
            api.submit_order(ticker,round(abs(int(num_of_shares))),'sell','market','day')
        except:
            pass 
            

def makeTrades(df, model):
    '''predicts on current state using pretrained model'''
    mappings = dict()
    i = 0
    # map data to index for model purposes
    for index, row in df.iterrows():
        mappings[i] = row['tic']
        i += 1

    print('mappings: ', mappings)
    
    # reload env to get current buying power (df, prices,ti, date) for model prediction 
    obs_trade = reset(df)

    actions, _states = model.predict(obs_trade)
    # obs_trade, rewards, dones, info = step(actions, i, mappings, state, reward)

    print('actions: ', actions)
    
    
    actions = actions * HMAX_NORMALIZE

    argsort_actions = np.argsort(actions)

    sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
    buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

    positions = api.list_positions()
    
    print('sell')
    for index in sell_index:
        #         stock ticker     num to sell for each ticker
        sell_stock(mappings[index],actions[index])
        
    
    print('buy')
    for index in buy_index:
        #daytrading_buying_power = 4 * (last_equity - last_maintenance_margin)
        
        buy_stock(mappings[index], int(actions[index]))
        print('cash', float(account.daytrading_buying_power))
        # print(4 * (float(account.last_equity) - float(account.last_maintenance_margin)))
        current_price = api.get_barset(mappings[index], 'day', limit=1)
        price = current_price[mappings[index]]
        cprice = price[-1].c
        cost = round((actions[index]*cprice)-0.5)
        print('cost', cost)
        
        # print(mappings[index], actions[index])
        

def reset(df, initial=True, previous_state=[]):
    '''According to A2c docs you need to reset trading env for predict on new data '''
    # grabs balance from alpaca
    account_balance = account.buying_power
    if initial:
        day = 0
        data = df.loc[day,:]
        turbulence = 0
        cost = 0
        trades = 0
        terminal = False
        #self.iteration=self.iteration
        rewards_memory = []
        #initiate state
        state = [account_balance] + \
                df.adjcp.values.tolist() + \
                [0]*STOCK_DIM + \
                df.macd.values.tolist() + \
                df.rsi.values.tolist() + \
                df.cci.values.tolist() + \
                df.adx.values.tolist()
    else:
        previous_total_asset = previous_state[0]+ \
        sum(np.array(previous_state[1:(STOCK_DIM+1)])*np.array(previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
        asset_memory = [previous_total_asset]
        #self.asset_memory = [self.previous_state[0]]
        day = 0
        data = df.loc[day,:]
        turbulence = 0
        cost = 0
        trades = 0
        terminal = False
        #self.iteration=iteration
        rewards_memory = []
        #initiate state
        #self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]
        #[0]*STOCK_DIM + \

        state = [account_balance] + \
                df.adjcp.values.tolist() + \
                [0]*STOCK_DIM + \
                df.macd.values.tolist() + \
                df.rsi.values.tolist() + \
                df.cci.values.tolist() + \
                df.adx.values.tolist()

    return state




if __name__ == "__main__":
    # tickers = get_highest_movers()
    #tickers = ['TSLA', 'CCL', 'ETSY', 'OXY', 'NCLH', 'FLS', 'SIVB', 'V', 'FANG', 'DG', 'MCHP', 'ENPH', 'MRO', 'BBY', 'CB', 'APA', 'DISCK', 'XRX', 'NKE', 'DISCA']
    # tickers = ['TSLA']
    
    tickers = ['PVH',
 'WBA',
 'VIAC',
 'DISCA',
 'KR',
 'ILMN',
 'EQIX',
 'DISCK',
 'RMD',
 'RL',
 'PENN',
 'ABMD',
 'ENPH',
 'LB',
 'ADSK',
 'KMB',
 'ZBRA',
 'IVZ',
 'BLK',
 'UA']
    
    print(tickers)

    model = load_model(tickers)
    
    print(model)
    
    
    # isOpen = alpaca.get_clock().is_open
    # while(not isOpen):
    #     clock = alpaca.get_clock()
    #     openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
    #     currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
    #     timeToOpen = int((openingTime - currTime) / 60)
    #     print(str(timeToOpen) + " minutes til market open.")
    #     time.sleep(60)
    #     isOpen = self.alpaca.get_clock().is_open

    data = preprocess_data(tickers, limit=2)
    data = data[(data.datadate >= data.datadate.max())]
    data = data.reset_index()
    data = data.drop(["index"], axis=1)
    data = data.fillna(method='ffill')
    # print(data)

    starttime = time.time()
    timeout = starttime + 60*60*7
    while time.time() <= timeout:
        print("starting iteration at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        data = preprocess_data(tickers, limit=2)
        data = data[(data.datadate >= data.datadate.max())]
        data = data.reset_index()
        data = data.drop(["index"], axis=1)
        data = data.fillna(method='ffill')
        makeTrades(data, model)
        time.sleep(60 - ((time.time() - starttime) % 60))
         



    # print('buying power', account.buying_power)
    # balance = float(account.equity) - float(account.last_equity)
    # print('account balance', balance)
    
    if account.trading_blocked:
        print('Account is currently restricted from trading.')

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(account.buying_power))
    print('equity: ',account.equity)
    # balance_change = float(account.equity) - float(account.last_equity)
    # print(f'Today\'s portfolio balance change: ${balance_change}')