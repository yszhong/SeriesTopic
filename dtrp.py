# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestRegressor
from sklearn import decomposition
import numpy
import pandas
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import time
import sys
import datetime
import logging

def read_djia_news():
    parser = lambda date: pandas.datetime.strptime(date, "%Y-%m-%d")
    data = pandas.read_csv("Combined_News_DJIA.csv", header=0, parse_dates=["Date"], date_parser=parser)
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    data = data.filter(regex=("Top.*")).apply(lambda x: "".join(str(x.values)), axis=1)
    return data

def generate_topic(data, num_topic=10):
    vector = CountVectorizer()
    vector.fit(data)
    vocab = vector.vocabulary_
    vector = CountVectorizer(stop_words="english", vocabulary = vocab.keys())
    X = vector.fit_transform(data)
    lda = decomposition.LatentDirichletAllocation(n_topics=num_topic, learning_method="online")
    for day in range(X.shape[0]-1, -1, -1):
        lda.partial_fit(X[day, :])
        doc_topic = lda.transform(X[day, :])
        alpha = sum(doc_topic) / len(doc_topic)
        eta = sum(doc_topic) / len(doc_topic)
        lda.set_params(doc_topic_prior=alpha, topic_word_prior=eta)
    doc_topic = lda.transform(X)
    doc_topic = pandas.DataFrame(doc_topic, index=data.index)
    return doc_topic

def read_djia():
    parser = lambda date: pandas.datetime.strptime(date, "%Y-%m-%d")
    data = pandas.read_csv("DJIA_table.csv", header=0, parse_dates=["Date"], date_parser=parser)
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    label = data["Adj Close"]
    return data, label

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
	return data, label

def generate_history(data, period=7):
    flag = True
    col = []
    for i in data.columns:
        for j in range(1, period+1):
            col.append(i + str(j))
            temp = data[i].diff(j).values
            temp = numpy.array([numpy.nan_to_num(dfer) for dfer in temp])
            temp = temp.reshape(-1, 1)
            if not flag:
                history = numpy.hstack((history, temp))
            if flag:
                history = temp
                flag = False
    history = numpy.nan_to_num(history)
    history = pandas.DataFrame(history, index = data.index, columns=col, copy=True)
    return history

def generate_timefeature(data):
	week = numpy.array([float(day.strftime("%w")) for day in data.index]).reshape(-1, 1)
	tf = pandas.DataFrame(week, index=data.index, columns=["WeekdayFlag"], copy=True)
	week = numpy.array([float(day.strftime("%j")) for day in data.index]).reshape(-1, 1)
	tf["DayNoFlag"] = pandas.DataFrame(week, index=data.index, copy=True)
	return tf

def rf_predict(data, label, period):
    ind = label.index
    data = data.values
    label = label.values
    rf = RandomForestRegressor()
    pred = numpy.array(numpy.zeros(len(label)), copy=True)
    for i in range(len(label)):
        if i > period:
            rf.fit(data[i-period:i, :], label[i-period:i])
            pred[i] = rf.predict(data[i, :])
    pred = pandas.Series(pred, index = ind)
    return pred

def combinepred(data1, data2, label, period):
	data = pandas.concat([data1,data2], axis=1)
	ind = label.index
	data = numpy.nan_to_num(data.values)
	label = label.values
	rf = RandomForestRegressor()
	pred = numpy.array(numpy.zeros(len(label)), copy=True)
	for i in range(len(label)):
		if i > period:
			rf.fit(data[i-period:i, :], label[i-period:i])
			pred[i] = rf.predict(data[i, :])
	pred = pandas.Series(pred, index=ind)
	return pred

def simple_weight(label, base_pred, ref_pred):
    label = label.values
    base_pred = base_pred.values
    ref_pred = ref_pred.values
    weight = numpy.zeros(len(label))
    for i in range(len(label)):
        if base_pred[i] == label[i]:
            weight[i] = 1
            continue
        weight[i] = abs((ref_pred[i]-label[i])/(base_pred[i]-label[i]))
        if weight[i] > 1:
            weight[i] = 1
        if weight[i] < 0:
            weight[i] = 0
    return weight

def rf_weights(data, label, weight, winsize, interv):
    ind = label.index
    data = data.values
    label = label.values
    rf = RandomForestRegressor()
    pred = numpy.array(numpy.zeros(len(label)), copy=True)
    for i in range(len(label)):
        period = winsize
        while i > period+1 and numpy.log(numpy.min(weight[i-period:i])) >= 0:
            period += 1
        if i > period:
            rf.fit(data[i-period:i, :], label[i-period:i], sample_weight=weight[i-period:i])
            pred[i] = rf.predict(data[i, :]*prweight(weight[i-period:i]))
    pred = pandas.Series(pred, index = ind)
    return pred

def prweight(trweight):
	for i in range(len(trweight)):
		trweight[i]  = 1 - trweight[i]
		trweight[i] *= 1 - (i + 1) / len(trweight)
		trweight[i] = 1 - trweight[i]
	w = numpy.mean(trweight)
	return w

def mapd(forecast, actual, interval):
	percent = 0.8
	L = percent * len(actual)
	actual = actual[L::interval+1]
	forecast = forecast[L::interval+1]
	m = 0
	for i in range(len(actual)):
		if forecast[i] > 0:
			m += abs(actual[i] - forecast[i]) / actual[i]
	m = 100 * m / len(actual)
	return m

def accuracy(forecast, actual, interval):
	up = 0
	down = 0
	zr = 0
	percent = 0.8
	L = percent * len(actual)
	actual = actual[L-1::interval+1]
	forecast = forecast[L-1::interval+1]
	for i in range(1, len(actual)):
		if abs(forecast[i]) > 0:
			down += 1
			if actual[i] == actual[i-1]:
				zr += 1
				continue
			fdiff = forecast[i]-forecast[i-1]
			adiff = actual[i]-actual[i-1]
			if fdiff/abs(fdiff) == adiff/abs(adiff):
				up += 1
	return float(up) / float(down) if float(down) > 0 else 0

def errorrate(forecast, actual, interval, ratio=0.1):
	up = 0
	down = 0
	zr = 0
	percent = 0.8
	L = percent * len(actual)
	actual = actual[L::interval+1]
	forecast = forecast[L::interval+1]
	for i in range(len(actual)):
		if abs(forecast[i]) > 0:
			down += 1
			fdiff = forecast[i]-actual[i]
			if abs(fdiff) / actual[i] <= ratio:
				up += 1
	m = float(up) / float(down) if float(down) > 0 else 0
	return m

def weight_propagation(base_data, ref_data, label, period=30, maxiter=100):
    ind = label.index
    base_pred = rf_predict(base_data, label, period)
    ref_pred = rf_predict(ref_data, label, period)
    deriv = []
    flag = True
    i = 1
    while flag:
        #print "Iteration " + str(i) + " Begins..."
        i += 1
        weights = simple_weight(label, base_pred, ref_pred)
        base_pred = rf_weights(base_data, label, weights, period)
        weights = simple_weight(label, ref_pred, base_pred)
        ref_pred = rf_weights(ref_data, label, weights, period)
        deriv.append(mapd(base_pred.values, label.values, interv))
        if sum(abs(label-base_pred)/label) < 1e-1 or i > maxiter:
            flag=False
        print mapd(base_pred.values, label.values, interv)
    return base_pred, deriv

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
						datefmt="%Y %b %d %H:%M:%S", filename="stocknewslog.log", filemode="a")
	console = logging.StreamHandler()  
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(message)s")  
	console.setFormatter(formatter)  
	logging.getLogger("").addHandler(console) 
	start = time.clock()
	winsize = 500
	ntopic = 100
	interv = 0
	rt=200
	if len(sys.argv) >= 2:
		if int(sys.argv[1]) > 1:
			winsize = int(sys.argv[1])
	if len(sys.argv) > 1:
		if int(sys.argv[2]) > 1:
			ntopic = int(sys.argv[2])
	if len(sys.argv) >= 4:
		if int(sys.argv[3]) >= 0:
			interv = int(sys.argv[3])
	if len(sys.argv) >= 5:
		if float(sys.argv[4]) >= 0:
			rt = float(sys.argv[4]) / 1000
	logging.info("Params(winsize,ntopic,interval,percent): "+str(winsize)+" "+str(ntopic)+" "+str(interv)+" "+str(rt))
	warnings.filterwarnings("ignore")
	
	data, label = read_djia()
	#data, label = read_nasdaq()
	history = generate_history(data, period=winsize)
	doc_topic = generate_topic(read_djia_news(), num_topic=ntopic)
	tf = generate_timefeature(data)
	
	base_pred = rf_predict(history, label, winsize)
	M = mapd(base_pred.values, label.values, interv)
	logging.info("History MAPD: "+str(M))
	M = errorrate(base_pred.values, label.values, interv, ratio=rt)
	logging.info("Error Ratio: "+str(M))
	M = accuracy(base_pred.values, label.values, interv)
	logging.info("Signal Accuracy: "+str(M))
	
	ref_pred = rf_predict(doc_topic, label, winsize)
	M = mapd(ref_pred.values, label.values, interv)
	logging.info("LDA MAPD: "+str(M))
	M = errorrate(ref_pred.values, label.values, interv, ratio=rt)
	logging.info("Error Ratio: "+str(M))
	M = accuracy(ref_pred.values, label.values, interv)
	logging.info("Signal Accuracy: "+str(M))
	
	com_pred = combinepred(history, tf, label, winsize)
	M = mapd(com_pred.values, label.values, interv)
	logging.info("Time&His MAPD: "+str(M))
	M = errorrate(com_pred.values, label.values, interv, ratio=rt)
	logging.info("Error Ratio: "+str(M))
	M = accuracy(com_pred.values, label.values, interv)
	logging.info("Signal Accuracy: "+str(M))
	com_pred = combinepred(doc_topic, tf, label, winsize)
	M = mapd(com_pred.values, label.values, interv)
	logging.info("Time&Doc MAPD: "+str(M))
	M = errorrate(com_pred.values, label.values, interv, ratio=rt)
	logging.info("Error Ratio: "+str(M))
	M = accuracy(com_pred.values, label.values, interv)
	logging.info("Signal Accuracy: "+str(M))
	com_pred = combinepred(history, doc_topic, label, winsize)
	M = mapd(com_pred.values, label.values, interv)
	logging.info("Doc&His MAPD: "+str(M))
	M = errorrate(com_pred.values, label.values, interv, ratio=rt)
	logging.info("Error Ratio: "+str(M))
	M = accuracy(com_pred.values, label.values, interv)
	logging.info("Signal Accuracy: "+str(M))
	
	weights = simple_weight(label, base_pred, ref_pred)
	new_pred = rf_weights(data, label, weights, winsize, interv)
	M = mapd(new_pred.values, label.values, interv)
	logging.info("Weighted MAPD: "+str(M))
	M = errorrate(new_pred.values, label.values, interv, ratio=rt)
	logging.info("Error Ratio: "+str(M))
	M = accuracy(new_pred.values, label.values, interv)
	logging.info("Signal Accuracy: "+str(M))
	#new_pred, M = weight_propagation(data, doc_topic, label, period=winsize, maxiter=5)
	end = time.clock()
	logging.info("Runtime: "+str(end-start)+" seconds")
	#print "Running Time: " + str(end - start) + " seconds"
