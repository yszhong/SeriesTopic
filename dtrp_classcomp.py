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
    word_dist = lda.components_
    return doc_topic, word_dist, vocab

def read_djia(thres):
    parser = lambda date: pandas.datetime.strptime(date, "%Y-%m-%d")
    data = pandas.read_csv("DJIA_table.csv", header=0, parse_dates=["Date"], date_parser=parser)
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    label = data["Adj Close"].values
    label = pandas.Series(label[::-1], index=data.index[::-1])
    label = bin_label(label, thres)
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
    imp = [[] for i in range(len(label))]
    for i in range(len(label)):
        if i > period:
            rf.fit(data[i-period:i, :], label[i-period:i])
            imp[i] = rf.feature_importances_.tolist()
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
	L = int(0.8 * len(actual))
	forecast = forecast
	actual = actual
	for i in range(0, len(actual), winsize):
		down += 1
		if actual[i] == 0:
			zr += 1
			continue
		if forecast[i] == actual[i]:
			up += 1
	accuracy = float(up) / float(down)
	return accuracy, zr

def showord(importance, words, vocabulary, label):
	for i in range(len(label)):
		imp = numpy.argsort(numpy.array(importance[i]))
		imp = imp[:10]
		topword = []
		w_up = []
		w_down = []
		for j in imp:
			mw = numpy.argsort(numpy.array(words[j, :]))
			if not len(topword):
				topword = mw[:10].reshape(1, -1)
			else:
				topword = numpy.concatenate(mw[:10].reshape(1, -1), axis=0)
		tops = [[item for item in vocabulary.keys() if vocabulary[item]==n] for n in topword]
		if label[i] == 1:
			w_up.append(tops)
		if label[i] == -1:
			w_down.append(tops)
	return w_up, w_down

def weight_propagation(base_data, ref_data, label, period=30, maxiter=100):
    ind = label.index
    base_pred, imp = rf_predict(base_data, label, period)
    ref_pred, imp = rf_predict(ref_data, label, period)
    deriv = []
    print "Ready"
    flag = True
    i = 1
    while flag:
        print "Iteration " + str(i+1) + " Begins..."
        i += 1
        weights = simple_weight(label, base_pred, ref_pred)
        base_pred = rf_weights(base_data, label, weights, period)
        weights = simple_weight(label, ref_pred, base_pred)
        ref_pred = rf_weights(ref_data, label, weights, period)
        d = numpy.array([mapd(base_pred.values, label.values, period), mapd(ref_pred.values, label.values, period)]).reshape(1, -1)
        if not len(deriv):
            deriv = d
        else:
            deriv = numpy.concatenate((deriv, d), axis=0)
        if sum(abs(label-base_pred)/label) < 1e-3 or i > maxiter:
            flag=False
    return base_pred, deriv

if __name__ == "__main__":
	start = time.clock()
	winsize = 7
	thres = 0
	if len(sys.argv) >= 2:
		if int(sys.argv[1]) > 1 and int(sys.argv[1]) < 1000:
			winsize = int(sys.argv[1])
	if len(sys.argv) >= 3:
		thres = int(sys.argv[2])
	print "Window Size is " + str(winsize)
	warnings.filterwarnings("ignore")
	data, label = read_djia(thres)
	history = generate_history(data, period=winsize)
	doc_topic, wrd, voc = generate_topic(read_djia_news(), num_topic=10)
	#print "Data Read"
	base_pred, imp = rf_predict(data, label, winsize)
	ref_pred, imp = rf_predict(doc_topic, label, winsize)
	#wu, wd = showord(imp, wrd, voc, label)
	#print wu, wd
	#print "Classifier Built"
	weights = simple_weight(label, base_pred, ref_pred)
	new_pred = rf_weights(data, label, weights, winsize)
	M, al = mapd(base_pred.values, label.values, winsize)
	print "History Accuracy: " + str(M)
	M, al = mapd(ref_pred.values, label.values, winsize)
	print "LDA Accuracy: " + str(M)
	M, al = mapd(new_pred.values, label.values, winsize)
	print "Weight Accuracy: " + str(M)
	#new_pred, M = weight_propagation(history, doc_topic, label, period=winsize, maxiter=100)
	#print "Propagation Accuracy: " + str(M)
	end = time.clock()
	#print "Running Time: " + str(end - start) + " seconds"
