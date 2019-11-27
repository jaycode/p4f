import pandas as pd
import os

API_PATH = """http://app.quotemedia.com/quotetools/getHistoryDownload.csv?&webmasterId=501&startDay={sd}&startMonth={sm}&startYear={sy}&endDay={ed}&endMonth={em}&endYear={ey}&isRanged=true&symbol={sym}"""

start_date, start_month, start_year = 2, 1, 2014
end_date, end_month, end_year = 31, 10, 2019

# In QuoteMedia, months start from 0, so we adjust these variables.
start_month = start_month - 1
end_month = end_month - 1

test_tickers = ['test1', 'test2']

def assert_history(get_history_fn):
    df = get_history_fn(API_PATH, start_date, start_month, start_year, end_date, end_month, end_year, 'MCD')
    assert df['adjclose'][0] == 196.70, "Data are not downloaded correctly. Please check if the API path and parameters are correct."
    assert ((os.path.isfile('MCD.csv'))), "Datasets must be stored as CSV files in the format of [ticker].csv."
    print("Passed!")

    
def assert_prices(get_prices_fn):
    df = get_prices_fn(API_PATH, start_date, start_month, start_year, end_date, end_month, end_year, test_tickers)    
    assert df.index[0].strftime('%Y-%m-%d') == '2019-11-23', "Index must be a 'datetime64[ns]' object, sorted from the earliest date."
    assert 'test1' in df.columns and 'test2' in df.columns, "DataFrame must contain the following columns: 'SPY', 'FB', 'AMZN', 'NFLX', 'GOOG'."
    assert df['test1'][0] == 10.0, "For non-date fields, they need to be filled with the values of 'adjclose' field."
    assert df.shape[0] == 4, "Rows must be properly combined with the union of both datasets on date as its index."
    print("Passed!")

    
def assert_return_percentages(get_return_percentages_fn):
    df = pd.DataFrame(columns=['value'], data=[1, 2, 4, 3])
    returns = get_return_percentages_fn(df)
    assert ((returns['value'][0] == 0.0) &
            (returns['value'][1] == 1.0) &
            (returns['value'][2] == 1.0) &
            (returns['value'][3] == -0.25)), "Incorrect return percentages."
    print("Passed!")

    
def assert_cumsum(get_cumsum_fn):
    df = pd.DataFrame(columns=['value'], data=[0.0, 1.0, 1.0, -0.25])
    returns = get_cumsum_fn(df)
    assert ((returns['value'][0] == 0.0) &
            (returns['value'][1] == 1.0) &
            (returns['value'][2] == 2.0) &
            (returns['value'][3] == 1.75)), "Incorrect cumsum results."
    print("Passed!")