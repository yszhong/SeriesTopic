#!/bin/bash

echo "Begins"
for ((i=1;i<=500;i+=1))
do
	python dtrp_classcomp.py $i
done
echo "End"
