# -*- coding:utf-8 -*-

import numpy
import pandas
import warnings
import time
import sys
import datetime
#import logging
from statsmodels.tsa.arima_model   import ARIMA

def read_djia():
    parser = lambda date: pandas.datetime.strptime(date, "%Y-%m-%d")
    data = pandas.read_csv("DJIA_table.csv", header=0, parse_dates=["Date"], date_parser=parser)
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    label = data["Adj Close"]
    return label

def read_nasdaq():
	parser = lambda date: pandas.datetime.strptime(date, "%Y/%m/%d")
	data = pandas.read_csv("NASDAQ.csv", header=0, parse_dates=["Date"], date_parser=parser)
	begin = datetime.date(2008, 8, 8)
	begin = len(data[pandas.to_datetime(data["Date"]) > begin])
	end = datetime.date(2016, 7, 1)
	end = len(data[pandas.to_datetime(data["Date"]) >= end])
	#print begin, data["Date"][begin]
	#print end, data["Date"][end]
	data = pandas.DataFrame(data.values[end:begin, :], columns=data.columns)
	data.index = data["Date"]
	data = data.drop(["Date"], axis=1)
	label = pandas.Series(data["Close"], index=data.index)
	label = label[end:begin]
	return label


def mapd(forecast, actual, interval):
	percent = 0.8
	L = percent * len(actual)
	actual = actual
	forecast = forecast
	m = 0
	for i in range(len(actual)):
		if forecast[i] > 0:
			m += abs(actual[i] - forecast[i]) / actual[i]
	m = 100 * m / len(actual)
	return m

def errorrate(forecast, actual, interval, ratio=0.02):
	up = 0
	down = 0
	zr = 0
	percent = 0.8
	L = percent * len(actual)
	actual = actual.values
	forecast = forecast.values
	for i in range(len(actual)):
		if abs(forecast[i]) > 0:
			down += 1
			fdiff = forecast[i]-actual[i]
			if abs(fdiff) / actual[i] <= ratio:
				up += 1
	m = float(up) / float(down)# if float(down) > 0 else 0
	return m

def arma(data, interval):
	interval += 1
	pred = numpy.zeros(len(data))
	ind = data.index
	i = 0
	data = data.values[::-1]
	while(i+interval<len(data)):
		serie = data[i:i+interval]
		i += interval
		model = ARIMA(serie, order=(0,1,1))
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		pred[i] = yhat
	pred = pandas.Series(pred[::-1], index=ind)
	return pred

if __name__ == "__main__":
	#logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
						#datefmt="%Y %b %d %H:%M:%S", filename="stocknewslog.log", filemode="a")
	#console = logging.StreamHandler()  
	#console.setLevel(logging.INFO)
	#formatter = logging.Formatter("%(message)s")  
	#console.setFormatter(formatter)  
	#logging.getLogger("").addHandler(console)
	interval = 10
	start = time.clock()

	warnings.filterwarnings("ignore")
	
	data = read_djia()
	#data = read_nasdaq()

	pred = arma(data, interval)
	print mapd(pred, data, interval)
	print errorrate(pred, data, interval)
	
	end = time.clock()
	#print end-start
	#logging.info("Runtime: "+str(end-start)+" seconds")
