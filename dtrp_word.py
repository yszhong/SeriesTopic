# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestRegressor
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
    wrd = lda.components_
    return doc_topic, wrd, vocab

def read_djia():
    parser = lambda date: pandas.datetime.strptime(date, "%Y-%m-%d")
    data = pandas.read_csv("DJIA_table.csv", header=0, parse_dates=["Date"], date_parser=parser)
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    label = data["Adj Close"]
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
    pred = numpy.array(label, copy=True)
    imp = []
    for i in range(0, len(label), period):
        if i > period:
            rf.fit(data[i-period:i, :], label[i-period:i])
            if len(imp):
                imp = numpy.concatenate((imp, rf.feature_importances_.reshape(1, -1)), axis=0)
            else:
                imp = rf.feature_importances_.reshape(1, -1)
            pred[i] = rf.predict(data[i, :])
    pred = pandas.Series(pred, index = ind)
    return pred, imp

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

def mapd(forecast, actual):
    return 100 * sum(abs(actual - forecast) / actual) / len(actual)

if __name__ == "__main__":
	start = time.clock()
	winsize = 14
	ntopic = 20
	if len(sys.argv) >= 2:
		if int(sys.argv[1]) > 0 and int(sys.argv[1]) < 1000:
			winsize = int(sys.argv[1])
	if len(sys.argv) >= 3:
		if int(sys.argv[2]) > 1:
			ntopic = int(sys.argv[2])
	#print "Window Size is " + str(winsize)
	#print "Amount of Topic: " + str(ntopic)
	warnings.filterwarnings("ignore")
	data, label = read_djia()
	history = generate_history(data, period=winsize)
	doc_topic, wrd, vocab = generate_topic(read_djia_news(), num_topic=ntopic)
	#print "Data Read"
	base_pred, imp = rf_predict(data, label, winsize)
	ref_pred, imp = rf_predict(doc_topic, label, winsize)
	#print "Regressor Built"
	weights = simple_weight(label, base_pred, ref_pred)
	new_pred = rf_weights(data, label, weights, winsize)
	marg = []
	for i in range(len(imp)):
		#if weights[i] < 1:
		marg.append(numpy.argmax(imp[i, :]))
	mt = max(marg, key=marg.count)
	mw = numpy.argmax(wrd[mt, :])
	print [x for x in vocab.keys() if vocab[x] == mw]
	M = mapd(new_pred.values, label.values)
	print "Mean Absolute Percentage Deviation: " + str(M)
	end = time.clock()
	#print "Running Time: " + str(end - start) + " seconds"
