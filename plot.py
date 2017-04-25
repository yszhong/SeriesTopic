# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:54:21 2017

@author: Zhong
"""

import re
from matplotlib import pyplot
import numpy
import pandas

s = open("../stocknewslog.log").read()
r = re.compile(r"\d\.\d+")
m = r.findall(s)
m = [float(i) for i in m]

partm = numpy.array(m).reshape(-1, 20)
clm = ["percent", "his-m", "his-e", "his-a", "l-m", "l-e", "l-a", "th-m", "th-e", "th-a", "td-m", "td-e", "td-a", "hd-m", "hd-e", "hd-a", "w-m", "w-e", "w-a", "time"]
data = pandas.DataFrame(partm, columns=clm)
rto = data.values[:, 3:19:3]
del clm, partm, m
#aprec = [0.084840319764, 0.888888888889]
splits = [1, 21, 31, 42, 59]
colorset = ["r", "b", "c", "y", "w", "g"]

for spli in range(len(splits)-1):
    fig, ax = pyplot.subplots(figsize=(6, 4))
    for c in range(rto.shape[1]):
        xax = numpy.arange(c/8, len(rto[splits[spli]:splits[spli+1], c]), 1)
        ax.bar(xax, rto[splits[spli]:splits[spli+1], c], width=0.1, color=colorset[c])
    pyplot.show()
del splits, spli, c, xax, colorset

f = open("longterm.txt").read()
r = re.compile(r"\d+\.\d+")
k = r.findall(f)
k = [float(i) for i in k]
k = [k[::2], k[1::2]]
k = numpy.array(k)
del f, r, s

fig, ax = pyplot.subplots(figsize=(6,4))
ax.plot(k[0])
ax.plot(k[1])
pyplot.show()