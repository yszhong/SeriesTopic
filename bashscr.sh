#!/bin/bash

echo "Start"
python dtrp.py 200 20 0 20
for ((i=2;i<=400;i+=20))
do
	python dtrp.py $i 20 0 20
done
for ((i=2;i<=20;i+=2))
do
	python dtrp.py 200 $i 0 20
done
for ((i=0;i<=200;i+=20))
do
	python dtrp.py 200 20 $i 20
done
for ((i=0;i<=80;i+=5))
do
	python dtrp.py 200 20 0 $i
done
echo "End"
