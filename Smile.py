#!/usr/bin/env python3
# Back test the Smile strategy: based on the trading strategy used in "The Dao
# of Capital" by Mark Spitznagel, in an account buy-and-hold an index, buy
# 2-month out OTM put with a certain proportion of the portfolio as an
# insurance and roll them over when they are 1-month out. The profit graph of
# this strategy looks like a smile where it goes up when the underlying index
# drops a lot or grows.

import torch
import argparse
import calendar
import csv
from datetime import datetime, timedelta
import json
import pandas as pd
import time
import sys
from optlib.gbs import (
    merton,
    euro_implied_vol
)

Debug = True

CONTRACT_SCALE = 100

# do not use $3000 to keep the gain/loss self contained
ANNUAL_CAPITAL_LOSS_DEDUCTION = 0

Msi = None
Msi_quartiles = None

Summary_fields = [
    'otmPct',           # distance of Put strike to stock price
    'putPct0',          # annual percent of NLV to buy put for low MSI Index
    'putPct1',
    'putPct2',
    'putPct3',          # annual percent of NLV to buy put for high MSI Index

    'annualReturn',
    'maxDrawdown',
    'taxRate',          # tax / finalNLV
    'commRate',         # commission / finalNLV
]

Monthly_fields = [
    'otmPct',           # distance of Put strike to stock price
    'putPct0',          # annual percent of NLV to buy put for low MSI Index
    'putPct1',
    'putPct2',
    'putPct3',          # annual percent of NLV to buy put for high MSI Index

    'date',

    'action',           # buy, sell
    'entType',          # stock, put

    'stkShares',        # negative means sell
    'stkOpenPrice',
    'stkPrice',

    'contracts',        # negative means sell
    'optOpenPrice',
    'optPrice',
    'strike',
    'expDate',          # expiration date of the option

    'stkGain',
    'optGain',          # gain (<0 means loss) of this action
    'totalShares',
    'totalStkGain',
    'totalOptGain',
    'totalTax',
    'totalComm',
    'totalGain',
    'optDelta',    # current delta of options / NLV
    'maxOptDelta',     # max delta of options (if stock drops to 0) / NLV
    'optNLV',           # total net liquidation value of all cur opt
    'NLV',              # total net liquidation value

    'buyHoldNLV',
    'buyHoldMaxDD',
    'buyHMaxDDPeak',
    'buyHMaxDDPkDate',

    'maxDrawdown',      # max draw down
    'maxDrawDPeak',
    'maxDDPeakDate',
    'curDrawdown',
    'curPeak',
    'curPeakDate',
]

def assert_close(a, b):
    if abs(a) < 0.0001 and abs(b) < 0.0001:
        return
    assert abs(a - b) / a < 0.001

def round_float(v, digits = -1):
    if not isinstance(v, float):
        return v
    if digits == -1:
        if v > 1000:
            return f'{int(v + 0.5):,}'
        elif v > 100:
            return int(v + 0.5)
        elif v > 10: # round to 1 digits
            return int(v * 10 + 0.5) / 10
        else: # round to 2 digits
            return int(v * 100 + 0.5) / 100
    else:
        n = 10**digits
        return int(v * n) / n

# make sure key exists in dictionary
def add_val(row, key, val):
    assert(key in row)
    if isinstance(val, float):
        row[key] = round_float(val)
    else:
        row[key] = val

def init_csv_row(param, fields):
    row = {key: 0 for key in fields}  # All keys initialized to 0
    row['otmPct'] = int(param.put_otm_percent * 100)
    for i in range(4):
        row[f'putPct{i}'] = \
            round_float(param.monthly_put_percent[i] * 100)
    return row

def add_summary_csv_row(param=None, annualReturn=0, maxDrawdown=0,
                        taxRate=0, commRate=0):
    row = init_csv_row(param, Summary_fields)
    add_val(row, 'annualReturn', f'{round_float(annualReturn)}%')
    add_val(row, 'maxDrawdown', f'{round_float(maxDrawdown)}%')
    add_val(row, 'taxRate', f'{round_float(taxRate)}%')
    add_val(row, 'commRate', f'{round_float(commRate)}%')
    param.fixed_param.summary_writer.writerow(row)

def add_monthly_csv_row(param=None, date='', action='buy', entType='stock',
                        stkShares=0, stkOpenPrice=0, stkPrice=0,
                        contracts=0, optOpenPrice=0, optPrice=0, expDate='',
                        strike=0,
                        stkGain=0, optGain=0,
                        stat=None, price_group=None):
    row = init_csv_row(param, Monthly_fields)
    add_val(row, 'date', date)
    add_val(row, 'action', action)
    add_val(row, 'entType', entType)

    add_val(row, 'stkShares', stkShares)
    add_val(row, 'stkOpenPrice', stkOpenPrice)
    add_val(row, 'stkPrice', stkPrice)

    assert contracts >= 0
    add_val(row, 'contracts', round_float(contracts, digits=0))
    add_val(row, 'optOpenPrice', optOpenPrice)
    add_val(row, 'optPrice', optPrice)
    add_val(row, 'strike', strike)
    add_val(row, 'expDate', expDate)

    add_val(row, 'stkGain', stkGain)
    add_val(row, 'optGain', optGain)

    add_val(row, 'totalShares', stat.total_shares())
    add_val(row, 'totalStkGain',
            stat.total_capital_gain - stat.total_put_capital_gain)
    add_val(row, 'totalOptGain', stat.total_put_capital_gain)
    add_val(row, 'totalTax', stat.total_tax)
    add_val(row, 'totalComm', stat.total_commission)
    add_val(row, 'totalGain', stat.total_capital_gain)

    add_val(row, 'optDelta', stat.opt_delta_rate())
    add_val(row, 'maxOptDelta', stat.max_opt_delta_rate())

    nlv, opt_nlv = stat.get_balance(price_group)
    add_val(row, 'optNLV', opt_nlv)
    add_val(row, 'NLV', nlv)

    add_val(row, 'buyHoldNLV', stat.buy_hold_nlv)
    add_val(row, 'buyHoldMaxDD', f'{round_float(stat.buy_hold_max_drawdown)}%')
    add_val(row, 'buyHMaxDDPeak', stat.buy_hold_max_drawdown_peak)
    add_val(row, 'buyHMaxDDPkDate', stat.buy_hold_max_drawdown_peak_date)

    add_val(row, 'maxDrawDPeak', stat.max_drawdown_peak)
    add_val(row, 'maxDDPeakDate', stat.max_drawdown_peak_date)
    add_val(row, 'maxDrawdown', f'{round_float(stat.max_drawdown)}%')
    if stat.cur_peak == 0:
        add_val(row, 'curDrawdown', 0)
    else:
        dd = 1 - nlv / stat.cur_peak
        add_val(row, 'curDrawdown', f'{round_float(dd)}%')
    add_val(row, 'curPeak', stat.cur_peak)
    add_val(row, 'curPeakDate', stat.cur_peak_date)

    param.fixed_param.monthly_writer.writerow(row)

def calculate_quartiles(data):
    # Sort the index values
    sorted_values = sorted(data)

    # Calculate quartile positions
    n = len(sorted_values)
    q1 = sorted_values[n // 4]  # 25th percentile
    q2 = sorted_values[n // 2]  # 50th percentile (median)
    q3 = sorted_values[(3 * n) // 4]  # 75th percentile

    return q1, q2, q3

def get_quartile(year):
    global Msi_quartiles
    if Msi_quartiles is None:
        # Extract index values from the dictionary
        index_val = [value['index'] for value in Msi.values()]
        Msi_quartiles = calculate_quartiles(index_val)

    value = Msi[year]['index']
    q1, q2, q3 = Msi_quartiles
    if value <= q1:
        return 0
    elif q1 < value <= q2:
        return 1
    elif q2 < value <= q3:
        return 2
    else:
        return 3

class FixedParam:
    def __init__(self,
                 initial_balance=1e6,
                 capital_gain_tax=0.371,
                 stock_comm_per_share=0.0035,
                 opt_comm_per_contract=0.065,
                 stock_slippage=0,
                 opt_slippage_percent=1,
                 monthly_writer=None,
                 summary_writer=None,
                 max_delta=0.5) -> None:
        self.initial_balance = initial_balance
        self.capital_gain_tax = capital_gain_tax
        self.stock_comm_per_share = stock_comm_per_share
        self.opt_comm_per_contract = opt_comm_per_contract
        self.stock_slippage = stock_slippage
        self.opt_slippage_percent = opt_slippage_percent / 100
        self.monthly_writer = monthly_writer
        self.summary_writer = summary_writer
        self.max_delta = max_delta

class Param:
    def __init__(self,
                 fixed_param=None,
                 put_otm_percent=30,
                 call_otm_percent=10,
                 monthly_put_percent=[],
                 call_percent=0) -> None:
        self.fixed_param = fixed_param
        self.put_otm_percent = put_otm_percent / 100
        self.call_otm_percent = call_otm_percent / 100
        self.monthly_put_percent = []
        self.call_percent = call_percent

        for i in range(4):
            if i >= len(monthly_put_percent):
                percent = monthly_put_percent[-1]
            else:
                percent = monthly_put_percent[i]
            self.monthly_put_percent.append(percent / 100)

class StockPosition:
    def __init__(self,
                 open_price = 0,
                 open_date = None,
                 num_shares = 0) -> None:
        self.open_price = open_price # includes per-share commission
        self.open_date = open_date
        self.num_shares = num_shares

class OptPosition:
    def __init__(self,
                 opt_type='p', # put or call
                 strike = 0,
                 exp_date = None,
                 num_contracts = 0,
                 open_date = None,
                 open_price = 0,
                 org_price = 0,
                 delta = 0,
                 implied_vol = 0) -> None:
        assert(opt_type in ['p', 'c'])
        self.opt_type = opt_type
        self.strike = strike
        self.exp_date = exp_date
        self.num_contracts = num_contracts
        self.open_date = open_date
        self.open_price = open_price # includes slippage & commission
        self.org_price = org_price # w/o slippage nor commission
        self.per_contract_delta = delta
        self.implied_vol = implied_vol

    def total_delta(self) -> float:
        return self.num_contracts * self.per_contract_delta * CONTRACT_SCALE

    def total_max_delta(self) -> float:
        # delta of put is negative
        return self.num_contracts * -1 * CONTRACT_SCALE

    def get_cost_basis(self) -> float:
        return self.open_price * self.num_contracts * CONTRACT_SCALE

    def __repr__(self):
        date = self.exp_date.strftime("%Y-%m-%d")
        return f'{int(self.strike)}{self.opt_type} exp={date}'

class PositionsAndStats:
    def __init__(self) -> None:
        self.all_data = None
        self.stocks = []
        self.cash = 0
        self.annual_capital_gain = 0
        self.total_tax = 0
        self.opt_list = []

        # stats
        self.initial_price = 0

        self.cur_peak = 0
        self.cur_peak_date = None

        self.max_drawdown_peak = None
        self.max_drawdown_peak_date = None
        self.max_drawdown = 0

        self.buy_hold_nlv = 0
        self.buy_hold_max_drawdown = 0
        self.buy_hold_max_drawdown_peak = 0
        self.buy_hold_max_drawdown_peak_date = 0
        self.buy_hold_cur_peak = 0
        self.buy_hold_cur_peak_date = 0

        self.total_capital_gain = 0
        self.total_put_capital_gain = 0
        self.total_commission = 0
        self.num_data_missing = 0
        self.num_strike_too_low = 0

    def print_fields(self):
        for attr in sorted(self.__dict__):
            if attr in ['all_data', 'stocks', 'opt_list']:
                continue
            print(f"{attr}: {round_float(self.__dict__[attr])}")

    def get_balance(self, price_group) -> (float, float):
        nlv = self.total_shares() * price_group['UnderlyingPrice'].iloc[0];
        opt_nlv = 0
        for opt in self.opt_list:
            put_info = \
                price_group[
                    (price_group['PutCall'] == 'p') &
                    (price_group['StrikePrice'] == opt.strike) &
                    (price_group['ExpirationDate'] == opt.exp_date)].\
                    sort_values(by='StrikePrice', ascending=False)
            if put_info.empty:
                self.num_data_missing += 1
                target_date = price_group['DataDate'].iloc[0]
                date_str = target_date.strftime("%Y-%m-%d")
                print(f'=== {date_str}: {opt.strike} Put expire on ' + \
                      f'{opt.exp_date} ' + \
                      f'missing, check prev day')
                date_range_start = target_date - pd.Timedelta(days=5)
                date_range_end = target_date + pd.Timedelta(days=0)
                data_slice = self.all_data[
                    (self.all_data['DataDate'] >= date_range_start) &
                    (self.all_data['DataDate'] <= date_range_end)
                    ]

                put_info = data_slice[
                    (data_slice['PutCall'] == 'p') &
                    (data_slice['ExpirationDate'] == opt.exp_date) &
                    (data_slice['StrikePrice'] == opt.strike)
                ].sort_values(by='DataDate', ascending=False)

                if put_info.empty:
                    print(f'=== {date_str}: {opt.strike} Put expire on ' + \
                          f'{opt.exp_date} ' + \
                          f'missing, check prev expire day')
                    new_exp_date = opt.exp_date - pd.Timedelta(days=1)
                    put_info = data_slice[
                        (data_slice['PutCall'] == 'p') &
                        (data_slice['ExpirationDate'] == new_exp_date) &
                        (data_slice['StrikePrice'] == opt.strike)
                    ]

                assert(not put_info.empty)

            mid = (put_info['BidPrice'].iloc[0] + \
                   put_info['AskPrice'].iloc[0]) / 2
            opt_nlv += mid * opt.num_contracts * CONTRACT_SCALE
        nlv += opt_nlv
        return nlv, opt_nlv

    def total_shares(self) -> float:
        total = sum(position.num_shares for position in self.stocks)
        return total

    def opt_delta_rate(self) -> float:
        delta = sum(opt.total_delta() for opt in self.opt_list)
        return delta / self.total_shares()

    def max_opt_delta_rate(self) -> float:
        delta = sum(opt.total_max_delta() for opt in self.opt_list)
        return delta / self.total_shares()

    def close_stock_positions(self, num_shares, close_price) -> None:
        assert(num_shares >= 0)

        # Sort the positions by cost basis in descending order
        sorted_positions = sorted(self.stocks,
                                  key=lambda x: x.open_price, reverse=True)

        shares_to_close = num_shares
        capital_gain = 0

        for position in sorted_positions:
            if shares_to_close <= 0:
                break

            shares_closed = min(position.num_shares, shares_to_close)
            capital_gain += (close_price - position.open_price) * \
                shares_closed
            position.num_shares -= shares_closed
            shares_to_close -= shares_closed

            # Remove the position if all its shares are closed
            if position.num_shares == 0:
                self.stocks.remove(position)

        self.total_capital_gain += capital_gain
        self.annual_capital_gain += capital_gain

        if num_shares == 0:
            avg_open_price = close_price
        else:
            avg_open_price = close_price - capital_gain / num_shares
        return capital_gain, avg_open_price

""" Index(['Symbol', 'ExpirationDate', 'AskPrice', 'AskSize', 'BidPrice',
       'BidSize', 'LastPrice', 'PutCall', 'StrikePrice', 'Volume',
       'openinterest', 'UnderlyingPrice', 'DataDate'], """

def third_friday(year, month):
    """Function to return the date of the third Friday of a given month."""
    first_day_of_month = datetime(year, month, 1)
    third_friday = first_day_of_month + \
        timedelta(days=((4-first_day_of_month.weekday()) % 7) + 14)
    return third_friday

def third_fri_after_2_mon(year, month):
    """Function to return the date of the third Friday of two months later."""
    # Adjusting for year wrap-around
    if month > 10:  # For November and December, go to next year
        year += 1
        # November becomes January, December becomes February
        month = (month + 1) % 12 + 1
    else:
        month += 2  # For other months, just add two

    return third_friday(year, month)

def get_sell_price(param, put_info):
    mid = (put_info['BidPrice'] + put_info['AskPrice']) / 2
    return mid * (1 - param.fixed_param.opt_slippage_percent)

def get_buy_price(param, opt_info) -> (float, float):
    mid = (opt_info['BidPrice'] + opt_info['AskPrice']) / 2
    return max(mid * (1 + param.fixed_param.opt_slippage_percent), 0), mid

def buy_initial_stock(param, date, stock_price, stat) -> None:
    # do not count stock slippage and commission for initial share
    open_price = stock_price # + param.fixed_param.stock_slippage + \
        # param.fixed_param.stock_comm_per_share

    num_shares = param.fixed_param.initial_balance / open_price

    stock_pos = StockPosition()
    stock_pos.open_price = open_price
    stock_pos.open_date = date
    stock_pos.num_shares = num_shares

    stat.stocks.append(stock_pos)

    stat.total_commission += num_shares * \
        param.fixed_param.stock_comm_per_share

def sell_old_put(param, date, group, stat) -> float:
    old_put_value = 0
    capital_gain = 0
    just_opened_opt = []
    stock_price = group['UnderlyingPrice'].iloc[0]
    date_str = date.strftime("%Y-%m-%d")
    for opt_pos in stat.opt_list:
        if opt_pos.open_date == date:
            just_opened_opt.append(opt_pos)
            continue
        put_info = \
            group[(group['PutCall'] == 'p') &
                  (group['StrikePrice'] == opt_pos.strike) &
                  (group['ExpirationDate'] == opt_pos.exp_date)]
        assert(len(put_info) == 1)
        put_sell_price = get_sell_price(param, put_info.iloc[0])
        old_put_value += \
            max((put_sell_price * CONTRACT_SCALE - \
                 param.fixed_param.opt_comm_per_contract) * \
                opt_pos.num_contracts,
                0)
        cur_capital_gain = old_put_value - opt_pos.get_cost_basis()
        capital_gain += cur_capital_gain
        stat.total_commission += \
            param.fixed_param.opt_comm_per_contract * opt_pos.num_contracts
        if Debug:
            nlv, opt_nlv = stat.get_balance(group)
            print(f'{date_str}: sell {opt_pos.num_contracts:.0f}' + \
                  f' at ${put_sell_price:.2f} {opt_pos}  ' + \
                  f'gain={capital_gain} stock={stock_price} ' + \
                  f'NLV={nlv:.0f} ' + \
                  f'put_gain={stat.total_put_capital_gain + capital_gain}')
        add_monthly_csv_row(
            param=param, date=date_str, action='sell',
            entType='put',
            stkPrice=stock_price,
            contracts=opt_pos.num_contracts,
            optOpenPrice=opt_pos.open_price,
            optPrice=put_sell_price,
            strike=opt_pos.strike,
            expDate=opt_pos.exp_date.strftime("%Y-%m-%d"),
            optGain=cur_capital_gain,
            stat=stat, price_group=group)

    stat.opt_list = just_opened_opt
    stat.total_capital_gain += capital_gain
    stat.total_put_capital_gain += capital_gain
    stat.annual_capital_gain += capital_gain

    return old_put_value

def delta_iv(param, date, stock_price, pos):
    days = (pos.exp_date - date).days
    time_to_expire_in_years = days / 365.25
    risk_free_rate = 0.0005 # TODO: use real treasury yield
    dividend_yield = 0.013 # TODO: use real dividend

    implied_vol = euro_implied_vol(
        'p', fs=stock_price, x=pos.strike, t=time_to_expire_in_years,
        r=risk_free_rate, q=dividend_yield, cp=pos.org_price)

    value, delta, gamma, theta, vega, rho = \
        merton('p', fs=stock_price, x=pos.strike, t=time_to_expire_in_years,
               r=risk_free_rate, q=dividend_yield, v=implied_vol)
    assert_close(value, pos.org_price)

    return delta, implied_vol

def open_new_option(param, date, group, stat, opt_type='p') -> float:
    assert opt_type in ['p', 'c']
    is_put = (opt_type == 'p')
    stock_price = group['UnderlyingPrice'].iloc[0]
    date_str = date.strftime("%Y-%m-%d")

    new_opt_pos = OptPosition(opt_type)
    new_opt_pos.exp_date = third_fri_after_2_mon(date.year, date.month)
    if is_put:
        strike = stock_price * (1 - param.put_otm_percent)
    else:
        strike = stock_price * (1 + param.call_otm_percent)
    new_opt_pos.open_date = date

    opt_info = \
        group[(group['PutCall'] == opt_type) &
              (group['StrikePrice'] <= strike if is_put else group['StrikePrice'] >= strike) &
              (group['ExpirationDate'] == new_opt_pos.exp_date)].\
              sort_values(by='StrikePrice', ascending=(not is_put))

    if opt_info.empty:
        new_exp_date = new_opt_pos.exp_date - pd.Timedelta(days=1)
        opt_info = \
            group[(group['PutCall'] == opt_type) &
                  (group['StrikePrice'] <= strike if is_put else group['StrikePrice'] >= strike) &
                  (group['ExpirationDate'] == new_exp_date)].\
                  sort_values(by='StrikePrice', ascending=(not is_put))
        if not opt_info.empty:
            print(f'=== {date_str}: no opt expire on ' + \
                  f'{new_opt_pos.exp_date} found expire on {new_exp_date}')
            new_opt_pos.exp_date = new_exp_date

    if opt_info.empty: # not found, skip
        stat.num_strike_too_low += 1
        return 0

    new_opt_info = opt_info.iloc[0]
    new_opt_pos.strike = new_opt_info['StrikePrice']

    new_opt_pos.open_price, new_opt_pos.org_price = \
        get_buy_price(param, new_opt_info)

    if new_opt_pos.open_price != 0:
        nlv, opt_nlv = stat.get_balance(group)

        new_opt_pos.per_contract_delta, new_opt_pos.implied_vol = \
            delta_iv(param, date, stock_price, new_opt_pos)

        if is_put:
            # delta of put is negative, convert it to positive
            max_contracts = stat.total_shares() * param.fixed_param.max_delta \
                / (abs(new_opt_pos.per_contract_delta) * CONTRACT_SCALE)

            put_percent = param.monthly_put_percent[get_quartile(date.year)]
            new_opt_pos.num_contracts = \
                min(max_contracts,
                    nlv * put_percent / \
                    (new_opt_pos.open_price + \
                     param.fixed_param.opt_comm_per_contract) / \
                    CONTRACT_SCALE)
            assert new_opt_pos.num_contracts >= 0
        else:
            max_call_contracts = nlv / CONTRACT_SCALE
            new_opt_pos.num_contracts = \
                -1 * int(max_call_contracts * (param.call_percent / 100))
            assert new_opt_pos.num_contracts <= 0 # sell calls, so must < 0

        new_opt_cost = \
            (new_opt_pos.open_price * CONTRACT_SCALE +
             param.fixed_param.opt_comm_per_contract) * \
             new_opt_pos.num_contracts

        stat.total_commission += \
            param.fixed_param.opt_comm_per_contract * new_opt_pos.num_contracts

        act = 'buy' if is_put else 'sell'
        if Debug:
            stock_price = group['UnderlyingPrice'].iloc[0]
            nlv, opt_nlv = stat.get_balance(group)
            print(f'{date_str}: {act} {abs(new_opt_pos.num_contracts):.0f} ' + \
                  f'at ${new_opt_pos.open_price:.2f} {new_opt_pos} ' + \
                  f'stock={stock_price} NLV={nlv:.0f} ' + \
                  f'put_gain={stat.total_put_capital_gain}')
        stat.opt_list.append(new_opt_pos)
        add_monthly_csv_row(
            param=param, date=date_str, action=act,
            entType='put' if is_put else 'call',
            stkPrice=stock_price,
            contracts=new_opt_pos.num_contracts,
            optOpenPrice=new_opt_pos.open_price,
            optPrice=new_opt_pos.open_price,
            strike=new_opt_pos.strike,
            expDate=new_opt_pos.exp_date.strftime("%Y-%m-%d"),
            stat=stat, price_group=group)
    else:
        new_opt_cost = 0
    return new_opt_cost

def buy_new_put(param, date, group, stat) -> float:
    return open_new_option(param, date, group, stat, opt_type='p')

def roll_over_option_pos(param, date, group, stat) -> None:
    stock_price = group['UnderlyingPrice'].iloc[0]

    # must buy new put before sell old put, so that the value of the old
    # put is considered when calculating NLV
    new_put_cost = buy_new_put(param, date, group, stat)

    old_put_value = sell_old_put(param, date, group, stat)

    net_balance = old_put_value - new_put_cost
    if net_balance > 0: # put generated profit, buy more shares
        fill_price = stock_price + \
            (param.fixed_param.stock_slippage + \
             param.fixed_param.stock_comm_per_share)
    else: # put costed money, sell some stock to make up
        fill_price = stock_price - \
            (param.fixed_param.stock_slippage + \
             param.fixed_param.stock_comm_per_share)

    shares = net_balance / fill_price

    date_str = date.strftime("%Y-%m-%d")
    if shares > 0: # open new stock position
        new_stock_pos = StockPosition()
        new_stock_pos.open_price = fill_price
        new_stock_pos.open_date = date
        new_stock_pos.num_shares = shares
        stat.stocks.append(new_stock_pos)
        if Debug:
            print(f'{date_str}: buy {shares:.1f} shares stock' + \
                  f' at {fill_price:.2f}')
        stat.total_commission += \
            param.fixed_param.stock_comm_per_share * abs(shares)
        add_monthly_csv_row(
            param=param, date=date_str, action='buy',
            entType='stock', stkShares=shares,
            stkOpenPrice=fill_price, stkPrice=fill_price, stat=stat,
            price_group=group)
    else: # close existing stock position
        gain, avg_open_price = stat.close_stock_positions(-shares, fill_price)
        stat.total_commission += \
            param.fixed_param.stock_comm_per_share * abs(shares)
        add_monthly_csv_row(
            param=param, date=date_str, action='sell',
            entType='stock', stkShares=shares, stkGain=gain,
            stkOpenPrice=avg_open_price, stkPrice=fill_price, stat=stat,
            price_group=group)
        if Debug:
            print(f'{date_str}: sell {shares:.1f} shares stock' + \
                  f' at {fill_price:.2f} gain:{gain:.1f}')

def year_end_handling(param, date, group, stat):
    stock_price = group['UnderlyingPrice'].iloc[0]
    last_year = date.year
    annual_gain = stat.annual_capital_gain
    if stat.annual_capital_gain < 0:
        # carry over loss to next year (reduce by $3,000 every year)
        if stat.annual_capital_gain <= \
           -1 * ANNUAL_CAPITAL_LOSS_DEDUCTION:
            stat.annual_capital_gain += \
                ANNUAL_CAPITAL_LOSS_DEDUCTION
        else:
            stat.annual_capital_gain = 0
    else: # deduct capital gain tax
        date_str = date.strftime("%Y-%m-%d")
        tax = annual_gain * param.fixed_param.capital_gain_tax
        fill_price = stock_price - param.fixed_param.stock_slippage - \
            param.fixed_param.stock_comm_per_share
        shares = tax / fill_price
        gain, avg_open_price = stat.close_stock_positions(shares, fill_price)
        stat.total_tax += tax
        add_monthly_csv_row(
            param=param, date=date_str, action='sell-pay-tax',
            entType='stock', stkShares=shares, stkGain=gain,
            stkOpenPrice=avg_open_price, stkPrice=fill_price, stat=stat,
            price_group=group)
        if Debug:
            print(f'{date_str}: Year End sell {shares:.1f} shares stock' + \
                  f' at {fill_price:.2f}')

def calc_drawdown(param, date_str, stat, group, cur_balance, stock_price):
   if stat.cur_peak < cur_balance:
       stat.cur_peak = cur_balance
       stat.cur_peak_date = date_str

   # Calculate maximum drawdown
   cur_drawdown = (1 - cur_balance / stat.cur_peak) * 100
   if stat.max_drawdown < cur_drawdown:
       stat.max_drawdown = cur_drawdown
       stat.max_drawdown_peak = stat.cur_peak
       stat.max_drawdown_peak_date = stat.cur_peak_date
       add_monthly_csv_row(
           param=param, date=date_str, action='maxDD',
           entType='', stkShares=0, stkGain=0,
           stkOpenPrice=stock_price, stkPrice=stock_price, stat=stat,
           price_group=group)

   prop = cur_balance / param.fixed_param.initial_balance
   if cur_drawdown > 90 or prop < 0.1:
       print(f'{date_str}: Strategy failed:' + \
             f'  drawdown: {cur_drawdown*100:.1f}%' + \
             f' balance: {prop * 100:.1f}%')
       return True
   else:
       return False

def process_buy_hold(param, date_str, stat, stock_price):
    # calculate buy-and-hold, dividend wasn't counted
    stat.buy_hold_nlv = stock_price / stat.initial_price * \
        param.fixed_param.initial_balance

    if stat.buy_hold_cur_peak < stat.buy_hold_nlv:
        stat.buy_hold_cur_peak = stat.buy_hold_nlv
        stat.buy_hold_cur_peak_date = date_str

    cur_buy_hold_drawdown = \
        (1 - stat.buy_hold_nlv / stat.buy_hold_cur_peak) * 100
    if stat.buy_hold_max_drawdown < cur_buy_hold_drawdown:
        stat.buy_hold_max_drawdown = cur_buy_hold_drawdown
        stat.buy_hold_max_drawdown_peak = \
            stat.buy_hold_cur_peak
        stat.buy_hold_max_drawdown_peak_date = \
            stat.buy_hold_cur_peak_date

# the main Smile strategy
def smile_strategy(df, param):
    global Debug

    stat = PositionsAndStats()
    stat.all_data = df
    last_year = None
    first_day = None
    last_day = None
    day_idx = 0
    stock_begin_price = 0
    stock_end_price = 0
    stat.buy_hold_nlv = param.fixed_param.initial_balance

    for date, group in df.groupby('DataDate'):
        date_str = date.strftime("%Y-%m-%d")
        stock_price = group['UnderlyingPrice'].iloc[0]
        stock_end_price = stock_price

        if first_day is None:
            assert(not date is None)
            first_day = date
            stat.initial_price = stock_price
        last_day = date

        if day_idx == 0: # first day
            stock_price = group['UnderlyingPrice'].iloc[0]
            buy_initial_stock(param, date, stock_price, stat)
            stock_begin_price = stock_price

        cur_balance, opt_nlv = stat.get_balance(group)

        failed = calc_drawdown(
            param, date_str, stat, group, cur_balance, stock_price)
        if failed:
            break

        process_buy_hold(param, date_str, stat, stock_price)

        # only roll put options on the 3rd Friday or on first trading day
        current_third_friday = third_friday(date.year, date.month)
        monthly = (day_idx == 0 or date == current_third_friday)

        if monthly:
            roll_over_option_pos(param, date, group, stat)

        if last_year is None:
            last_year = date.year

        # pay tax at year end and update balance for the new year
        if last_year != date.year:
            year_end_handling(param, date, group, stat)
            last_year = date.year

        day_idx += 1

    # Performance Metrics Calculation
    total_years = (last_day - first_day).days / 365.25
    annual_return = \
        ((cur_balance / param.fixed_param.initial_balance) ** \
         (1 / total_years) - 1)*100

    buy_hold_annual_return = \
        ((stock_end_price / stock_begin_price) ** (1 / total_years) - 1)*100

    add_summary_csv_row(param, annualReturn=annual_return,
                        maxDrawdown=stat.max_drawdown,
                        taxRate=stat.total_tax / cur_balance * 100,
                        commRate=stat.total_commission / cur_balance * 100)

    return {
        'AllStats': stat,
        'EndNLV': cur_balance,
        'TotalYears': total_years,
        'BuyHoldAnnualReturn': buy_hold_annual_return,
        'AnnualReturn': annual_return,
        'MaxDrawdown': stat.max_drawdown,
        'TotalCommission': stat.total_commission,
        'TotalTaxPaid': stat.total_tax,
        'NumDataMissing': stat.num_data_missing,
    }

def test_all_params(params, df):
    for param in params:
        result = smile_strategy(df, param)
        for k, v in result.items():
            if isinstance(v, float):
                print(f'{k}: {round_float(v)}')
            elif isinstance(v, PositionsAndStats):
                print('--- all stats start ---')
                v.print_fields()
                print('--- all stats end ---')

def main():
    parser = argparse.ArgumentParser(
        description="Specify optional input/output arguments")

    parser.add_argument(
        '-C', '--call-otm', type=str, default='6,11,1',
        help="OTM Call percent, 'otmPercent' or 'start,stop,step'. " + \
        "Stop is inclusive. Default 6,11,1")

    parser.add_argument(
        '-c', '--call-percent', type=int, default="0",
        help="Maximum percent of call options to sell. 80 means sell 8 calls for 1000 shares")

    parser.add_argument(
        '-D', '--max-delta', default=0.5,
        help="Maximum delta of options to buy/sell, avoid buying/selling too expensive ones")

    parser.add_argument(
        '-d', '--debug', action='store_true',
        help="Turn on debug output. Default off")

    parser.add_argument(
        '-i', '--input-path', default='SPY_50000.csv',
        help="Input file name (default: 'SPY_50000.csv')")

    parser.add_argument(
        '-m', '--msindex', default='msi.json',
        help="MSI data in json format (default: 'msi.json')")

    parser.add_argument(
        '-o', '--output-path', default='out.csv',
        help="Output path (default: 'out.csv')")

    parser.add_argument(
        '-P', '--put-otm', type=str, default='30,30,1',
        help="OTM Put percent, 'otmPercent' or 'start,stop,step'. " + \
        "Stop is inclusive. Default 30,30,1")

    parser.add_argument(
        '-p', '--put-percent-of-msi-quartile',
        type=str, nargs='*',
        default=[
            '0.0,0.0,0.0,0.0',
            '0.0,0.0,0.3,0.5',
            '0.1,0.2,0.3,0.5',
            '0.5,0.5,0.5,0.5'],
        help='Up to 4 numbers for MSI quartile 0, 1, 2, 3.' + \
        ' If less than 4, repeat the last number. E.g., 1,2 means 1,2,2,2' + \
        ' Lower is safer. More than one set can be given.')

    parser.add_argument(
        '-s', '--summary-output-path', default='sum.csv',
        help="Output path of summary for each param (default: 'sum.csv')")

    args = parser.parse_args()

    global Debug
    Debug = args.debug

    global Msi
    f = open(args.msindex)
    jsonMsi = json.load(f)
    f.close()
    Msi = {pd.to_datetime(entry['date']).year:
           {k: v for k, v in entry.items() if k != 'date'}
           for entry in jsonMsi}

    #for year in Msi:
    #    print(year, get_quartile(year))

    all_put_percent_args = args.put_percent_of_msi_quartile
    assert(len(all_put_percent_args) > 0)

    print(f'Output file: {args.output_path} {args.summary_output_path}')

    fout_monthly = open(args.output_path, 'w', encoding='UTF8', newline='')
    monthly_writer = csv.DictWriter(fout_monthly, fieldnames=Monthly_fields)
    monthly_writer.writeheader()

    fout_summary = open(args.summary_output_path, 'w', encoding='UTF8',
                        newline='')
    summary_writer = csv.DictWriter(fout_summary, fieldnames=Summary_fields)
    summary_writer.writeheader()

    fixed_param = FixedParam(monthly_writer = monthly_writer,
                             summary_writer = summary_writer,
                             initial_balance = 3 * 1000 * 1000,
                             max_delta = args.max_delta)

    params = []

    put_otm_info = args.put_otm.split(',')
    if len(put_otm_info) == 3:
        put_otm_start = int(put_otm_info[0])
        put_otm_stop = int(put_otm_info[1])
        put_otm_step = int(put_otm_info[2])
    elif len(put_otm_info) == 1:
        put_otm_start = int(put_otm_info[0])
        put_otm_stop = put_otm_start
        put_otm_step = 1
    else:
        print(f'"-put-otm" must be 1 or 3 comma separated numbers: {args.put_otm}')
        sys.exit(1)

    call_otm_info = args.call_otm.split(',')
    if len(call_otm_info) == 3:
        call_otm_start = int(call_otm_info[0])
        call_otm_stop = int(call_otm_info[1])
        call_otm_step = int(call_otm_info[2])
    elif len(call_otm_info) == 1:
        call_otm_start = int(call_otm_info[0])
        call_otm_stop = call_otm_start
        call_otm_step = 1
    else:
        print(f'"-C" must be 1 or 3 comma separated numbers: {args.call_otm}')
        sys.exit(1)

    all_monthly_put_percent = []
    for one_set_str in all_put_percent_args:
        one_set = one_set_str.split(',')
        if len(one_set) > 4:
            print(f'More than 4 numbers of -p not allowed: {one_set_str}')
            sys.exit(1)
        put_args = []
        for i in range(4):
            if len(one_set) <= i:
                put_args.append(float(one_set[-1]))
            else:
                put_args.append(float(one_set[i]))
        all_monthly_put_percent.append(put_args)

    for monthly_put_percent in all_monthly_put_percent:
        for put_otm in range(put_otm_start, put_otm_stop + put_otm_step,
                         put_otm_step):
            for call_otm in range(call_otm_start, call_otm_stop + call_otm_step,
                              call_otm_step):
                param = Param(
                    fixed_param = fixed_param,
                    put_otm_percent = put_otm,
                    call_otm_percent = call_otm,
                    monthly_put_percent = monthly_put_percent,
                    call_percent = args.call_percent)
                params.append(param)
                all_percent = set(monthly_put_percent)
                if len(all_percent) == 1 and next(iter(all_percent)) == 0:
                    break

    df = pd.read_csv(args.input_path, encoding='utf-8',
                     skipinitialspace=True)

    # debug format problem, very slow
    #for index, row in df.iterrows():
    #    try:
    #        # Attempt to convert the ExpirationDate to datetime
    #        pd.to_datetime(row['ExpirationDate'])
    #    except Exception as e:
    #        # If a ValueError is raised, print the problematic row and its index
    #        print(f"Error {e} in row {index}: {row['ExpirationDate']}")
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])
    df['DataDate'] = pd.to_datetime(df['DataDate'])
    df.sort_values(by='DataDate', inplace=True)

    print(f'{len(df.index)} rows read')

    test_all_params(params, df)

    fout_monthly.close()
    fout_summary.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    sec = time.time() - start_time
    print(f"--- {sec:.0f} seconds ---")
