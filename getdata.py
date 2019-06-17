from pandas_datareader import data as web
import datetime as dt
import math
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model

start = dt.datetime(2008, 6, 1)
end = dt.datetime(2018, 6, 1)

df = web.get_data_yahoo('AAPL', start=start, end=end)
#print(df.tail())

def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out); #creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size) #cross validation 

    response = [X_train, X_test , Y_train, Y_test, X_lately]
    return response


forecast_col = 'Adj Close'#choosing which column to forecast
forecast_out = 14 #how far to forecast 
test_size = 0.3; #the size of my test set

X_train, X_test, Y_train, Y_test , X_lately = prepare_data(df, forecast_col, forecast_out, test_size); #calling the method were the cross validation and data preperation is in

learner = linear_model.LinearRegression(); #initializing linear regression model

learner.fit(X_train,Y_train); #training the linear regression model
score = learner.score(X_test,Y_test); #testing the linear regression model

forecast = learner.predict(X_lately); #set that will contain the forecasted data


def workdays(d, count, excluded=(6, 7)):
    days = []
    numberofday = 0
    while numberofday < count:
        if d.isoweekday() not in excluded:
            days.append(d)
            numberofday += 1 
        d += dt.timedelta(days=1)
    return days

end_date = end + dt.timedelta(days=1)
list_date = workdays(end_date, 14)
date_list = [dt.datetime.strftime(i, "%Y-%m-%d") for i in list_date]

predicted = pd.DataFrame(forecast, date_list, columns =['Predicted Adj Close']) 

p_start = list_date[0]
p_end = list_date[len(list_date) - 1]
new_df = web.get_data_yahoo('AAPL', start=p_start, end=p_end)
print(new_df)

predicted['Actual Adj Close'] = new_df['Adj Close']
print(predicted)

