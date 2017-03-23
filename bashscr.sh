#!/bin/bash

echo "Begins"
for i in {0..100}
do
	python dtrp_clword.py 7 $i
done
echo "End"
