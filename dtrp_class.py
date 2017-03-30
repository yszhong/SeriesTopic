# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
import numpy
import pandas
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import time
import sys

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

def read_djia(thres):
    parser = lambda date: pandas.datetime.strptime(date, "%Y-%m-%d")
    data = pandas.read_csv("DJIA_table.csv", header=0, parse_dates=["Date"], date_parser=parser)
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    label = data["Adj Close"].values
    label = pandas.Series(label[::-1], index=data.index[::-1])
    label = bin_label(label, thres)
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
	label = bin_label(label, thres)
	print label
	return data, label

def bin_label(label, thres):
	ind = label.index
	label = label.diff(1).values
	for i in range(len(label)):
		if abs(label[i]) < thres:
			label[i] = 0
	label = label / abs(label)
	label = numpy.nan_to_num(label)
	label = pandas.Series(label, index=ind)
	return label

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
    rf = RandomForestClassifier()
    pred = numpy.array(label, copy=True)
    for i in range(0, len(label), period):
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
    rf = RandomForestClassifier()
    pred = numpy.array(label, copy=True)
    for i in range(0, len(label), winsize):
        period = 1
        while numpy.log(weight[i-period]) >= 0 and i > period+1:
            period += 1
        if period >= winsize:
            rf.fit(data[i-period:i, :], label[i-period:i], sample_weight=weight[i-period:i])
            pred[i] = rf.predict(data[i, :])
    pred = pandas.Series(pred, index = ind)
    return pred

def mapd(forecast, actual, winsize):
	up = 0
	down = 0
	zr = 0
	for i in range(0, len(actual), winsize):
		if forecast[i] == actual[i]:
			up += 1
		down += 1
		if actual[i] == 0:
			zr += 1
	accuracy = float(up) / float(down)
	return accuracy, zr

if __name__ == "__main__":
	start = time.clock()
	winsize = 7
	thres = 10
	if len(sys.argv) >= 2:
		if int(sys.argv[1]) > 1 and int(sys.argv[1]) < 1900:
			winsize = int(sys.argv[1])
	if len(sys.argv) >= 3:
		thres = int(sys.argv[2])
	print "Window Size is " + str(winsize)
	warnings.filterwarnings("ignore")
	data, label = read_djia(thres)
	data, label = read_djia(thres)
	history = generate_history(data, period=winsize)
	doc_topic = generate_topic(read_djia_news(), num_topic=10)
	#print "Data Read"
	base_pred = rf_predict(data, label, winsize)
	ref_pred = rf_predict(doc_topic, label, winsize)
	#print "Classifier Built"
	weights = simple_weight(label, base_pred, ref_pred)
	new_pred = rf_weights(data, label, weights, winsize)
	M, al = mapd(new_pred.values, label.values, winsize)
	print "Accuracy: " + str(M)
	print "Zero Values: " + str(al)
	end = time.clock()
	#print "Running Time: " + str(end - start) + " seconds"
