#!/bin/bash

echo "Start"
for ((i=1;i<=1000;i+=10))
do
	python dtrp.py $i 10 1
done
echo "End"
