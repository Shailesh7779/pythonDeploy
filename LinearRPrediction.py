import quandl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from flask import Flask, jsonify

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

quandl.ApiConfig.api_key = "xBLitG1foBj4M2YBXPxy"
app = Flask(__name__)


@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"


BIT_COIN_CSV_URL = 'http://www.quandl.com/api/v1/datasets/BCHARTS/KRAKENEUR.csv?api_key=xBLitG1foBj4M2YBXPxy'
df = pd.read_csv(BIT_COIN_CSV_URL, header=0, index_col='Date', parse_dates=True)
df1 = pd.read_csv(BIT_COIN_CSV_URL, header=0, index_col='Date', parse_dates=True)
# df = quandl.get("WIKI/BOM539678")
# df = df[['Adj. Close']]
df = df.reindex(index=df.index[::-1])
df1 = df1.reindex(index=df1.index[::-1])
# print(df)


#df_past30 = df[df.last_valid_index()-pd.DateOffset(30, 'D'):]
df1 = df1[df1.drop(df1.tail(30).index, inplace=True):]
# df_new = df[df.drop(df.tail(30).index, inplace=True):]# drop last n rows

# print(df1)
df = df[['Weighted Price']]
df1 = df1[['Weighted Price']]
forecast_out = int(30)  # predicting 30 days into future

forecast_out30 = int(30)  # predicting 30 days into future
# df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)  #  label column with data shifted 30 units up
df['Prediction'] = df[['Weighted Price']].shift(-forecast_out)  # label column with data shifted 30 units up
df1['Prediction'] = df1[['Weighted Price']].shift(-forecast_out30)  # label column with data shifted 30 units up

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:]  # set X_forecast equal to last 30
X = X[:-forecast_out]  # remove last 30 from X
y = np.array(df['Prediction'])
y = y[:-forecast_out]
#----------------30-06------------------
J = np.array(df1.drop(['Prediction'], 1))
J = preprocessing.scale(J)
J_forecast = J[-forecast_out30:]  # set X_forecast equal to last 30
J = J[:-forecast_out]  # remove last 30 from X
k = np.array(df1['Prediction'])
k = k[:-forecast_out30]


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
J_train, J_test, k_train, k_test = cross_validation.train_test_split(J, k, test_size=0.2)
# Training
clf = LinearRegression()
clf1 = LinearRegression()
clf.fit(X_train, y_train)
clf1.fit(J_train, k_train)
# Testing
confidence = clf.score(X_test, y_test)
confidence1 = clf1.score(J_test, k_test)
print("confidence: ", confidence1)
forecast_prediction = clf.predict(X_forecast)
forecast_prediction1 = clf1.predict(J_forecast)
print(forecast_prediction)
print(forecast_prediction1)

df['Forecast'] = np.nan
df1['Forecast1'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
date_array = []

for i in forecast_prediction:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    date_array.append(float(next_unix))

    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    # list[next_date.date()]

# -------------------------------2/7/2018-----------------------

last_date1 = df1.iloc[-1].name
last_unix1 = last_date1.timestamp()
one_day1 = 86400
next_unix1 = last_unix1 + one_day1
date_array1 = []


for i in forecast_prediction1:
    next_date1 = datetime.datetime.fromtimestamp(next_unix1)
    next_unix1 += one_day1
    date_array1.append(float(next_unix1))
    
    df1.loc[next_date1] = [np.nan for _ in range(len(df1.columns)-1)] + [i]
print(date_array1)
# df['Weighted Price'].plot()
# df1['Forecast1'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()
#
# # df['Weighted Price'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()
# print(df1)

# POST
@app.route('/api', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """

    return jsonify({'prediction': list(forecast_prediction), 'dates': date_array, 'prediction30': list(forecast_prediction1), 'dates30': date_array1})


# running web app in local machine
if __name__ == '__main__':
    app.run('www.abraca-dabra.com/',debug=True)