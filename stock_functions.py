import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.model_selection import RandomizedSearchCV

import sqlalchemy as db

import pickle
import requests

import starfishX as sx
from pytrends.request import TrendReq
import pandas_ta as pta
from tqdm.notebook import tqdm 
from statsmodels.tsa.stattools import grangercausalitytests
from prophet import Prophet

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

matplotlib.rc('font', family='Ayuthaya')

def cal_stock_ema(stock_symbol, df_stock_data):
    '''
    Calculate the stock EMA data overtime for the different set periods (Exponential moving average)
    
    Input:
    - stock data (equivalent to CLOSE column of stock data)
    
    Output:
    - Dataframe with the stock price, and EMA with different periods (10, 30, 50)
    
    '''
        
    # Create dataframe for the finalised output EMA data from stock
    ema_data = pd.concat([
                        df_stock_data.ewm(span=10,adjust=False).mean().rename(stock_symbol + '_ema10')
                        ,df_stock_data.ewm(span=30,adjust=False).mean().rename(stock_symbol + '_ema30')
                        ,df_stock_data.ewm(span=50,adjust=False).mean().rename(stock_symbol + '_ema50')
                    ],axis=1)
    
    return ema_data

def cal_stock_ma(stock_symbol, df_stock_data):
    '''
    Calculate the stock MA data overtime for the different set periods (Moving average)
    
    Input:
    - stock data (equivalent to CLOSE column of stock data)
    
    Output:
    - Dataframe with the stock price, and MA with different periods (10, 30, 50, 100)
    
    '''
        
    # Create dataframe for the finalised output MA data from stock
    ma_data = pd.concat([
                        df_stock_data.rolling(10).mean().rename(stock_symbol + '_ma10')
                        ,df_stock_data.rolling(30).mean().rename(stock_symbol + '_ma30')
                        ,df_stock_data.rolling(50).mean().rename(stock_symbol + '_ma50')
                        ,df_stock_data.rolling(100).mean().rename(stock_symbol + '_ma100')
                    ],axis=1)
    
    return ma_data

def cal_stock_rsi(stock_symbol, df_stock_data):
    '''
    Calculate the stock RSI data overtime for the different set periods (Relative Strength Index)
    
    Input:
    - stock data (equivalent to CLOSE column of stock data)
    
    Output:
    - Dataframe with the stock price, and RSI with different periods (2, 6, 14, 30)
    
    '''
        
    # Create dataframe for the finalised output RSI data from stock
    rsi_data = pd.concat([
                        pta.rsi(df_stock_data, length = 2).rename(stock_symbol + '_rsi2')
                        ,pta.rsi(df_stock_data, length = 6).rename(stock_symbol + '_rsi6')
                        ,pta.rsi(df_stock_data, length = 14).rename(stock_symbol + '_rsi14')
                        ,pta.rsi(df_stock_data, length = 30).rename(stock_symbol + '_rsi30')
                    ],axis=1)
    
    return rsi_data

def min_max_scaler_transform(df_raw):
    '''
    Function to transform the input dataframe to be normalised between 1 and 0.
    This is used to be able to normalise the data from different sources and format so that it is on the same scale.
    
    For example, comparing the trend between data of different dimensions such as stock price and google trend interest.
    By transforming and normalising them, we are able to see them on the same scale (ie, stock price may be different value
    to the google interest trends value and the scalability of the covid19 Thailand statistics as the rate of change
    may be different so it is wise to transform them and use the rescaled values.)

    Input:
    - Dataframe
    
    Output:
    - Transformed dataframe scaled in each column
    - min_max_scaler (used to revert transform the data back to its original scale)


    Example:
    - min_max_scaler_transform(df) -> df_transformed, min_max_scaler
    - min_max_scaler_inverse_transform(df_transformed, min_max_scaler) -> original_df
    
    Note:
    - Require usage with the min_max_scaler_inverse_transform function
    - Save the min_max_scaler output information to be able to reverse it appropriately
    
    '''
    
    # Create a MinMaxScaler and fit_transform them to normalise the data
    min_max_scaler = MinMaxScaler()
    df_transformed = min_max_scaler.fit_transform(df_raw)
    df_transformed = pd.DataFrame(df_transformed)
    
    # Rename the columns so they can be identified
    df_transformed.columns = df_raw.columns
    df_transformed.index = df_raw.index
    
    return df_transformed, min_max_scaler

def min_max_scaler_inverse_transform(df_transformed, min_max_scaler):
    '''
    Function to inverse the transformation of the dataframe

    Input:
    - Dataframe
    - min_max_scaler
    
    Output:
    - Inverse transform of dataframe to original 

    Example:
    - min_max_scaler_transform(df) -> df_transformed, min_max_scaler
    - min_max_scaler_inverse_transform(df_transformed, min_max_scaler) -> original_df
    
    '''
    
    # Inverse transform the normalised or minmaxscaled data
    df_reverted = min_max_scaler.inverse_transform(df_transformed)
    df_reverted = pd.DataFrame(df_reverted)
    
    # Rename the columns so they can be identified
    df_reverted.columns = df_transformed.columns
    df_reverted.index = df_transformed.index
    
    return df_reverted

def get_all_sources_stock_data(stock_symbol = 'DELTA'):
    '''
    Function to get all the functions we did above to create big data table for raw data frame
    Then output different tables that we can use for machine learning and train, test, pipelines, prediction
    
    Function includes information from
    - SET stock information
    - SET index information
    - Similar stocks based on same sector
    - SET stock MA, EMA, RSI
    - Google trends interese keywords
    - Similar stocks based on SET100 correlation
    - Covid19 Thailand stats
    
    Here, we are cleaning NaN data with forward/backward fill because there are cases when the data is not available
    in the weekends and the data extracted does not align in days (some return in weekly, some weekdays, etc).
    Forward/backward fill is used to fill in the data gaps for these parts.
        
    Input:
    - stock_symbol
    
    Process:
    - Retrieve the data from the used sources
    - Clean the data format
    - Merge the different data sources to the same dataframe
    - Forward/Backward to fill the missing data
    
    Output:
    - Dataframe with index date, and data feature columns for stock information
    
    Note
    - No plots or visualisation is made in this function. Please refer to the other 
    
    '''
    
    ########## Initiate stock data and fetch stock details ##########
    # Capitalise symbol so base usage standard
    stock_symbol = stock_symbol.upper()
    start_date = '20200101'
    
    # Check whether the stock is in SET100 or not
    is_stock_in_set100 = sx.getMemberOfIndex(sx.indexMarket.SET100)['symbol'].isin([stock_symbol]).any()
    
    # Check whether the input stock is in SET100. If not, then we reject them.
    if not is_stock_in_set100:
        print('Stock:', stock_symbol, ': is not in SET100. Please try again')
        return 0

    # Create a file name to save the SET stock data and its path
    set_sector_filename = '20210901_set_stock_industry'
    set_sector_csv = Path('./' + set_sector_filename + '.csv')

    # If the file exists, then read the csv data, else fetch latest data
    if set_sector_csv.is_file():
        print('File already exist....')

    else:
        # Fetch latest stock name and industry data (takes a long time to load)
        df_all_stock = sx.listSecurities(industry=True) # Getting name and market
        df_all_stock.to_csv(set_sector_filename +'.csv')
        print('File does not exist....created new csv file')

    # Read the stock name and its industry from file as it is faster
    df_all_stock = pd.read_csv(set_sector_csv)
    
    # Load stock historical data
    df_stock = sx.loadHistData(stock_symbol,start=start_date,OHLC=True, Volume=True)
    
    ########## Calculate SET50 SET100 index ##########
    # Get SET50 and SET100 index value
    df_index_50 = sx.marketview(sx.indexMarket.SET50,start=start_date)['index'].rename('set50_index')
    df_index_100 = sx.marketview(sx.indexMarket.SET100,start=start_date)['index'].rename('set100_index')
    df_set50_set100_index = pd.concat([df_index_50, df_index_100], axis=1)
    
    ########## Calculate foreign SET trade ##########
    # Fetch the foreign SET trade buy/sell value
    df_foreign_SET_trade = sx.marketViewForeignTrade(sx.indexMarket.SET, viewplot=True, start=start_date)

    # Filter based on the start date (as there is a bug with the library)
    df_foreign_SET_trade = df_foreign_SET_trade[['SET.BUY','SET.SELL']][df_foreign_SET_trade.index >= start_date].add_prefix('foreign_')
    
    ########## Calculate similar stocks by industry ##########
    # Create dataframe for similar stock stats
    df_sim_stock_industry = pd.DataFrame()

    # Fetch the stock list data for similar stock symbol from same sector
    stocks_list = sx.listStockInSector(stock_symbol)

    # Loop through the similar stocks
    for stock_name_in_sector in tqdm(stocks_list['symbol']):
        # Print stock data that is being fetched
        print('Fetching stock data in sector:', stock_name_in_sector)
        
        # Try catch loop to loop through and if some stock fails, then skil
        try:
            # Get the historical price of a specific stock
            temp_stock_data = sx.loadHistData(stock_name_in_sector,start=start_date,OHLC=True, Volume=True)

            # Get EMA, MA, RSI data
            ema_data = cal_stock_ema(stock_name_in_sector, temp_stock_data['CLOSE'])
            ma_data = cal_stock_ma(stock_name_in_sector, temp_stock_data['CLOSE'])
            rsi_data = cal_stock_rsi(stock_name_in_sector, temp_stock_data['CLOSE'])

            # Update suffix of the stock symbol
            temp_stock_data = temp_stock_data.add_prefix(stock_name_in_sector + '_')

            # Concatenate the dataframe to create a final output
            df_sim_stock_industry = pd.concat([df_sim_stock_industry, temp_stock_data, ema_data, ma_data, rsi_data],axis=1)

        except Exception as e:
            print(e)

    ########## Calculate similar stocks by SET100 correlation ##########
    df_stock_set100 = pd.DataFrame()
    
    # Get SET100 stock symbol list
    set100_symbol_list = sx.getMemberOfIndex(sx.indexMarket.SET100)

    # Loop through the list of set100 stock and create dataframe with all stock close
    for set100_stock_symbol in tqdm(set100_symbol_list['symbol']):
        # Print stock data that is being fetched
        print('Fetching stock data set100:', set100_stock_symbol)
        
        # Try and catch statement to skip those we cannot fetch data
        try:

            # Load the stock data for the specific stock from SET100
            stock_data = sx.loadHistData(set100_stock_symbol,start=start_date,OHLC=False, Volume=False)

            # Concat to the dataframe for SET100 stocks
            df_stock_set100 = pd.concat([df_stock_set100, stock_data],axis=1)

        except Exception as e:
            print(e)

    # Define the temp dataframe for stock correlation
    df_sim_stock_SET100_corr = pd.DataFrame()

    # Find the stocks with highest correlation
    corr_values = abs(df_stock_set100.corr())[stock_symbol].sort_values(ascending=False)[1:6]

    # For the stocks with top correlation value, get the data and create new dataframe
    for corr_stock in tqdm(corr_values.index.values):
        # Print stock data that is being fetched
        print('Fetching stock data correlation:', corr_stock)

        # Fetch the data for the correlated stock
        stock_data = sx.loadHistData(corr_stock,start=start_date,OHLC=False, Volume=False)

        # Concat the stock data together so that we can identify them
        df_sim_stock_SET100_corr = pd.concat([df_sim_stock_SET100_corr, stock_data], axis=1)

    # Rename and add suffix as identifier
    df_sim_stock_SET100_corr = df_sim_stock_SET100_corr.add_suffix('_SET100_corr')
    
    ########## Calculate Google trends interest with stock ##########
    # Setup empty dataframe to use for concat data
    df_stock_gg_trends = pd.DataFrame()

    # Instantiate pytrend class for setup
    pytrend = TrendReq()

    # Build the keyword payload
    pytrend.build_payload(kw_list=[stock_symbol],geo='TH', timeframe='today 5-y')

    # Identify the interest overtime with the inputted keyword, and filter unwanted case
    interest_data = pytrend.interest_over_time()[:-1].drop('isPartial',axis=1)
    interest_data = interest_data[interest_data.index >= start_date]

    # Instantiate pytrend class for setup
    pytrend2 = TrendReq()
    pytrend3 = TrendReq()

    # Get the name of the sector and industry for search
    sector_industry_data = df_all_stock[df_all_stock['symbol'] == stock_symbol]['industry'].str.split('/',expand=True).rename(columns={0: "sector", 1: "industry"})
    sector_name = sector_industry_data['sector'].values[0]
    industry_name = sector_industry_data['industry'].values[0]

    # Build the keyword payload
    pytrend2.build_payload(kw_list=[sector_name],geo='TH', timeframe='today 5-y')    
    pytrend3.build_payload(kw_list=[industry_name],geo='TH', timeframe='today 5-y')    

    sector_interest = pytrend2.interest_over_time()[:-1].drop('isPartial',axis=1)
    sector_interest = sector_interest[sector_interest.index >= start_date]
    industry_interest = pytrend3.interest_over_time()[:-1].drop('isPartial',axis=1)
    industry_interest = industry_interest[industry_interest.index >= start_date]

    # Instantiate pytrend class for related keywords
    pytrend4 = TrendReq()
    pytrend5 = TrendReq()

    # Identify the related query terms
    related_queries1 = pytrend.related_queries()[stock_symbol]['top'].iloc[0]['query']
    related_queries2 = pytrend.related_queries()[stock_symbol]['top'].iloc[1]['query']

    # Build the keyword payload
    pytrend4.build_payload(kw_list=[related_queries1],geo='TH')
    pytrend5.build_payload(kw_list=[related_queries2],geo='TH')

    # Get keyword interest data for related queries
    related_query_interest_data1 = pytrend4.interest_over_time()[:-1].drop('isPartial',axis=1)
    related_query_interest_data1 = related_query_interest_data1[related_query_interest_data1.index >= start_date]
    related_query_interest_data2 = pytrend5.interest_over_time()[:-1].drop('isPartial',axis=1)
    related_query_interest_data2 = related_query_interest_data2[related_query_interest_data2.index >= start_date]

    # Concat and create dataframe for output Google interest trends
    df_stock_gg_trends = pd.concat([interest_data, sector_interest, industry_interest, related_query_interest_data1, related_query_interest_data2], axis=1)

    # Rename column names to be static so it does not change overtime for prediction
    df_stock_gg_trends.columns = [stock_symbol + '_gg_trends'
                                     , 'sector_gg_interest'
                                     , 'industry_gg_interest'
                                     , 'top1_related_gg_interest'
                                     , 'top2_related_gg_interest']


    ########## Calculate COVID19 Thailand stats ##########
    # Fetch request statistics information from Thailand website for stats
    covid19_th_raw = requests.get('https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-all')

    # Create dataframe extracting the data from json dictionary
    df_covid19_th = pd.DataFrame(covid19_th_raw.json())

    # Set index as Date
    df_covid19_th = df_covid19_th.set_index('txn_date')

    # Create information to filter based on start date of the covid stats
    df_covid19_th = df_covid19_th[df_covid19_th.index >= start_date]

    # Drop the last data update column as we do not want to use it
    df_covid19_th.drop('update_date',axis=1, inplace=True)

    # Remove today's data as may not be fully accurate yet for usage
    df_covid19_th = df_covid19_th[:-1]

    # Scope the data for only the new and total case for usage
    df_covid19_th = df_covid19_th[['new_case','total_case']]

    ########## Merge data to create single dataframe source ##########
    # Create merged dataframe (without covid19 stats data)
    df_merge = pd.concat([df_stock.add_prefix(stock_symbol + '_')[stock_symbol + '_CLOSE']
                          , df_set50_set100_index
                          , df_foreign_SET_trade 
                          , df_sim_stock_industry
                          , df_sim_stock_SET100_corr
                          , df_stock_gg_trends], axis=1)
    
    # Update the index type to be string so that it matches covid19 dataframe index type
    df_temp_merge = df_merge.reset_index()
    df_temp_merge['index'] = df_temp_merge.reset_index()['index'].dt.strftime('%Y-%m-%d')
    df_temp_merge = df_temp_merge.set_index('index')
    
    # Merge all data sources with covid19 stats
    df_merge_all = pd.concat([df_temp_merge, df_covid19_th],axis=1)
    
    # Sort the index so that it is sorted by date properly
    df_merge_all.sort_index(inplace=True)

    # Forward fill and backward fill for the data since there are gaps in the data.
    df_final = df_merge_all.ffill(axis=0).bfill(axis=0)
    
    # Drop any duplicate columns
    df_final = df_final.loc[:,~df_final.columns.duplicated()]

    return df_final
    
def get_granger_causality_lag(df, granger_period = 10):
    '''
    Function to take in the dataframe, analyse the granger causality between the column with the rest of the features using the first column as its base.
    Then output data with the delayed lag time.
    
    Granger Causality testing
    - The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another.
    - Granger causality is a statistical concept of causality that is based on prediction. 
    - According to Granger causality, if a signal X1 "Granger-causes" (or "G-causes") a signal X2, then past values of X1 should contain information that helps predict X2 above and beyond the information contained in past values of X2 alone.
    - Based on f test case sum square regression (error)
    - The Sum of Squared regression is the sum of the differences between the predicted value and the mean of the dependent variable.

    p value
    - p value is the statistical significance. 
    - More than 0.05 = there is no significance so no causality from these metrics
    - Therefore, we take p value is less than 0.05

    f test 
    - The F-test is to test whether or not a group of variables has an effect on y, meaning we are to test if these variables are jointly significant.
    - f test to see how significant based on others and square error for each lag time
    - https://tien89.wordpress.com/2010/03/18/testing-multiple-linear-restrictions-the-f-test/
    
    Input:
    - DataFrame of the stock information to process
    - granger_period - For granger causality to calculate
    
    Process:
    - Loop through data to find granger causality for each pair with the first column (with interested data)
    - Identify the most appropriate delay/lag with granger causality testing

    Output:
    - Dataframe of stock transformed data with granger causality
    - Dataframe of grangercausality lag for each column
    - Dataframe of grouped lag value data by their columns for ease of reference
    
    '''
    
    # Get data source for all stocks with its features
    df_data = df.copy()
    
    # Create dataframe for granger transformed information
    df_granger_transformed = df.copy()
    
    # Create temp dataframe to hold the granger data
    df_granger = pd.DataFrame()
    
    # Create list for temp of lag_data
    lag_data = []
    
    # Loop through each data column and then evaluate the granger causality between each pair with first column of interested stock
    for idx, col in tqdm(enumerate(df_data.columns)):

        # Skip the data of the first column which is our interested stock
        if idx == 0:
            continue
        
        try:
            
            # Calculate the granger causality of each pair with the first column
            df_granger_raw = grangercausalitytests(pd.concat([df_data.iloc[:,0],df_data.iloc[:,idx]],axis=1), granger_period, verbose=False)

            # Create temporary dataframe for holding the granger results
            df_granger_results = pd.DataFrame()

            # Loop in the granger results data in order to sort and find the most appropriate lag period
            for num, item in df_granger_raw.items():

                # Extract the data and transpose it 
                df_ftest = pd.DataFrame(item[0]['ssr_ftest']).transpose()

                # Returned results (Fvalue, pValue, df_denom, df_num) for each of the results
                df_granger_results = pd.concat([df_granger_results, df_ftest],axis=0)

            # Rename the column data for ease of access
            df_granger_results.columns = ['Fvalue','pvalue','df_denom','df_num']

            # Sort by the pvalue less than 0.05 for significance, and choose the highest Fvalue
            lag_array = df_granger_results[df_granger_results['pvalue'] < 0.05].sort_values(by='Fvalue', ascending=False).head(1).df_num.values

            # Set initial lag value of grangercausality to 0
            lag_value = 0

            # If the value is not zero, then get the lag. otherwise lag_value is zero
            if len(lag_array) != 0:
                lag_value = lag_array[0]

            # Append the data to the lag data list
            lag_data.append((lag_value, col))

            # Shift the data in the column according to the granger causality test shift
            df_granger_transformed.iloc[:,idx] = df_data.iloc[:,idx].shift(int(lag_value))

            # Backfill the data from the shift as it results in null
            df_granger_transformed.bfill(axis=0, inplace=True)
        
        except Exception as e:
            print(e)

    # Rename thee dataframe for appropriate column naming
    lag_data = pd.DataFrame(lag_data).rename(columns={0 : 'lag_value', 1 : 'col'})
    
    # Filter out the ones with lag data more than 1
    lag_data = lag_data[lag_data['lag_value'] > 1].sort_values(by='lag_value', ascending=False)
    
    # Filter out only the columns which have granger causality affect
    df_granger_transformed = df_granger_transformed[lag_data['col']]

    # Group the data by its count so ease of reference usage
    lag_data_col = lag_data.groupby('lag_value').col.apply(list).reset_index().sort_values(by='lag_value',ascending=False)
    
    return df_granger_transformed, lag_data, lag_data_col

def create_train_test_predict_split_with_granger(df, train_size = 200, test_size = 50):
    '''
    Function to create train-test, train-predict dataset split for the stock source data input.
    Outcome of this function is to return the list of data with the train/test/predict split based on the different lag values.
    
    Input:
    - df = Dataframe of the stock_all_source data (with the first column being the stock of interest)
    - train_size = Training size for the split
    - test_size = Testing size for the split
    
    Output:
    - train_test_list_array = List of train_test split with (lag_value, train_x, test_x, train_y, test_y) dataset
    - train_pred_list_array = List of train_pred split with (lag_value, train_pred_x, test_pred_x, train_pred_y) dataset
    
    '''
    
    # Create an array for train-test and train-predict set for the different granger periods
    train_test_list_array = []
    train_pred_list_array = []

    # Create a list for the accumulated column for storage and reference
    col_accum = []

    # Create the stock data based on the data used 
    df_data = df.copy()

    # Set the stock data that we are testing for y value (assume first column data)
    df_stock_data = df.iloc[:,0]

    # Set the index to datetime format so we can shift them
    df_data.index = pd.to_datetime(df_data.index)
        
    # Create granger transformed and lag data
    df_granger_transformed, lag_data, lag_data_col = get_granger_causality_lag(df)

    # Loop for each lag_data_column 
    for idx, data in lag_data_col.iterrows():

        # For each data in each column, we append the accumulated column data
        # For this case, the most lag should be in all receding data to use
        for col in data['col']:
            col_accum.append(col)

        # Extract the lag_value data for the stock
        lag_value = int(data['lag_value'])

        ########## Section to create the dataframe for train-test results ##########
        # Create the dataset for train_x
        df_train = df_granger_transformed[col_accum][-(train_size + test_size):-test_size]

        # Create the dataset for test_x
        df_test = df_granger_transformed[col_accum][-test_size:]

        # Create the dataset for train_y 
        df_stock_data_train = df_stock_data[-(train_size + test_size):-test_size]

        # Create the dataset for test_y
        df_stock_data_test = df_stock_data[-test_size:]

        # Append the data into the list for the specific lag_value
        train_test_list_array.append([lag_value, df_train, df_test, df_stock_data_train, df_stock_data_test])

        ########## Section to create the dataframe for train-predict results ##########
        # Shift the feature data for granger causality lag value
        shifted_data = df_data[col_accum].shift(periods=lag_value,freq='D')

        # Create the dataset for train_x (prediction)
        df_train_pred = shifted_data[col_accum][-(train_size + lag_value):-lag_value]

        # Create the dataset for test_x (prediction)
        df_test_pred = shifted_data[col_accum][-lag_value:]

        # Create the dataset for train_y (prediction)
        df_stock_data_train_pred = df_stock_data[-train_size:]

        # Store the data in train prediction list dataset
        train_pred_list_array.append([lag_value, df_train_pred, df_test_pred, df_stock_data_train_pred])

    return train_test_list_array, train_pred_list_array

def clean_train_test_data(train_test, stock_symbol = 'DELTA_CLOSE', data_index = 2):
    '''
    Function to clean the train and test data that was prepared before modelling.
    
    Input:
    - train_test dataframe from create_train_test_predict_split_with_granger
    
    Output:
    - train_x, train_y, test_x, test_y = Data for modelling ujsage
    - df_test_y = For plotting purposes
    
    '''


    # Define the data point that we want to use (which lag value data)
    data_point = data_index

    # Define the stock symbol column name with _CLOSE
    stock_symbol_close = stock_symbol

    # Define the train and test datasets
    train_x = train_test[data_point][1]
    test_x = train_test[data_point][2]
    train_y = train_test[data_point][3]
    test_y = train_test[data_point][4]

    # Scale all the features so that it is in the same scale for testing
    df_transformed1, scaler1 = min_max_scaler_transform(pd.concat([train_x,train_y],axis=1))

    # Drop and split the transformed dataframe out appropriately
    train_x = df_transformed1.drop(stock_symbol_close,axis=1)
    train_y = df_transformed1[stock_symbol_close]

    # Scale the test results using the same scaler as above
    temp_scale_test = pd.DataFrame(scaler1.transform(pd.concat([test_x,test_y], axis=1)))
    temp_scale_test.columns = np.concatenate((test_x.columns, [stock_symbol_close]))

    # Drop the interest data in the dataframe that we do not want
    test_x = temp_scale_test.drop(stock_symbol_close,axis=1)
    test_y = temp_scale_test[stock_symbol_close]

    # Reset the index data for the date for each as lost when scale
    test_x.index = train_test[data_point][2].index
    test_y.index = train_test[data_point][4].index

    # Create dataframe for test_y for plotting and testing performance purposes
    df_test_y = test_y.reset_index().drop('index',axis=1)
    df_test_y.index = test_y.index

    return train_x, train_y, test_x, test_y, df_test_y

def clean_train_pred_data(train_pred, stock_symbol = 'DELTA_CLOSE', data_index = 2):
    '''
    Function to clean the train and pred data that was prepared before modelling.
    Only difference here is that there is no test_y data.
    
    Input:
    - train_pred dataframe from create_train_test_predict_split_with_granger
    
    Output:
    - train_x, train_y, test_x = Data for modelling usage
    - scaler = to rescale back to original for getting price values
    
    '''
    
    # Define the data point that we want to use (which lag value data)
    data_point = data_index

    # Define the stock symbol column name with _CLOSE
    stock_symbol_close = stock_symbol
    
    # Define the train and test datasets
    train_x = train_pred[data_point][1]
    test_x = train_pred[data_point][2]
    train_y = train_pred[data_point][3]

    # Scale all the features so that it is in the same scale for testing
    df_transformed1, scaler0 = min_max_scaler_transform(pd.concat([train_x,train_y],axis=1))
    
    # Min max scale the data
    train_x, scaler1 = min_max_scaler_transform(train_x)
    train_y, scaler2 = min_max_scaler_transform(pd.DataFrame(train_y))

    # Scale the test results using the same scaler as above
    temp_scale_test = pd.DataFrame(scaler1.transform(test_x))
    
    test_x.index = train_pred[data_point][2].index
    
    return train_x, train_y, test_x, scaler2