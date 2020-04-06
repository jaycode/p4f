import pandas as pd
import os
import math

test_tickers = ['test1', 'test2']

def assert_history(get_history_fn):
    df = get_history_fn('MCD')
    assert df['adjclose'][0] == 196.70, "Data are not downloaded correctly. Please check if the API path and parameters are correct."
    assert ((os.path.isfile('MCD.csv'))), "Datasets must be stored as CSV files in the format of [ticker].csv."
    print("Passed!")

    
def assert_prices(get_prices_fn):
    df = get_prices_fn(test_tickers)    
    assert df.index[0].strftime('%Y-%m-%d') == '2019-11-23', "Index must be a 'datetime64[ns]' object, sorted from the earliest date."
    assert 'test1' in df.columns and 'test2' in df.columns, "DataFrame must contain the following columns: 'SPY', 'FB', 'AMZN', 'NFLX', 'GOOG'."
    assert df['test1'][0] == 10.0, "For non-date fields, they need to be filled with the values of 'adjclose' field."
    assert df.shape[0] == 4, "Rows must be properly combined with the union of both datasets on date as its index."
    print("Passed!")

    
def assert_return_percentages(get_return_percentages_fn):
    df = pd.DataFrame(columns=['value'], data=[100, 101, 111, 83.25])
    returns = get_return_percentages_fn(df)
    assert ((math.isclose(round(returns['value'][0],7), 0.0)) &
            (math.isclose(round(returns['value'][1],7), 0.01)) &
            (math.isclose(round(returns['value'][2],7), 0.0990099)) &
            (math.isclose(round(returns['value'][3],7), -0.25))), "Incorrect return percentages."
    print("Passed!")

    
def assert_cumulative(get_cum_fn):
    df = pd.DataFrame(columns=['value'], data=[0.0, 0.01, 0.0990099, -0.25])
    returns = get_cum_fn(df)
    assert ((math.isclose(round(returns['value'][0],7), 0.0)) &
            (math.isclose(round(returns['value'][1],7), 0.01)) &
            (math.isclose(round(returns['value'][2],7), 0.11)) &
            (math.isclose(round(returns['value'][3],7), -0.1675))), "Incorrect cumulative results."
    print("Passed!")