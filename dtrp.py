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

def read_djia_news():
    data = pandas.read_csv("Combined_News_DJIA.csv", header=0)
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
    doc_topic = pandas.DataFrame(doc_topic)
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

def generate_history(data, period = 7):
    flag = True
    col = []
    for i in data.columns:
        for j in range(1, period+1):
            col.append(i + str(j))
            temp = data[i].diff(j).values.reshape(-1, 1)
            if not flag:
                history = numpy.hstack((history, temp))
            if flag:
                history = temp
                flag = False
    history = numpy.nan_to_num(history)
    history = pandas.DataFrame(history, index = data.index, columns=col, copy=True)
    return history

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

def rf_weights(data, label, weight, winsize):
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
		trweight[i] *= 1 - i / len(trweight)
		trweight[i] = 1 - trweight[i]
	w = numpy.mean(trweight)
	return w

def mapd(forecast, actual, interval):
	percent = 0.8
	L = percent * len(actual)
	actual = actual[L:]
	forecast = forecast[L:]
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
	actual = actual[L-1:]
	forecast = forecast[L-1:]
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
	return float(up) / float(down)

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
    return base_pred, deriv

if __name__ == "__main__":
	start = time.clock()
	winsize = 50
	ntopic = 20
	interv = 0
	if len(sys.argv) >= 2:
		if int(sys.argv[1]) > 1 and int(sys.argv[1]) < 1900:
			winsize = int(sys.argv[1])
	if len(sys.argv) >= 3:
		if int(sys.argv[2]) > 1:
			ntopic = int(sys.argv[2])
	if len(sys.argv) >= 4:
		if int(sys.argv[3]) >= 0:
			interv = int(sys.argv[3])
	print "Params:", winsize, ntopic, interv
	#print "Amount of Topic: " + str(ntopic)
	warnings.filterwarnings("ignore")
	data, label = read_djia()
	#data, label = read_nasdaq()
	history = generate_history(data, period=winsize)
	doc_topic = generate_topic(read_djia_news(), num_topic=ntopic)
	#print "Data Read"
	base_pred = rf_predict(data, label, winsize)
	M = mapd(base_pred.values, label.values, interv)
	print "History:", M
	M = accuracy(base_pred.values, label.values, interv)
	print "Acc:", M
	ref_pred = rf_predict(doc_topic, label, winsize)
	M = mapd(ref_pred.values, label.values, interv)
	print "LDA:", M
	M = accuracy(ref_pred.values, label.values, interv)
	print "Acc:", M
	#print "Regressor Built"
	weights = simple_weight(label, base_pred, ref_pred)
	new_pred = rf_weights(data, label, weights, winsize)
	M = mapd(new_pred.values, label.values, interv)
	print "Weighted:", M
	M = accuracy(new_pred.values, label.values, interv)
	print "Acc:", M
	#new_pred, M = weight_propagation(data, doc_topic, label, period=winsize, maxiter=5)
	#print "Propagation:", M
	end = time.clock()
	#print "Running Time: " + str(end - start) + " seconds"
