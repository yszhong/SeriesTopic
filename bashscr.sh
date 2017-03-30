#!/bin/bash

echo "Start"
for ((i=1;i<=100;i+=1))
do
	python dtrp_classcomp.py 100 $i 0
done
echo "End"
