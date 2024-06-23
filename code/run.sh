#!/bin/bash

variance=(0.9999 0.999 0.99)
kvals=(1 2 3 4 5 10 11 12 13 14 15 16 17 18 19 20 1000 1250 1500 1700 3000 3600 3700 3971)
model=('none' 'rf' 'rbfSVM')
epsilon=('epsilon50' 'epsilon5' 'epsilon10' 'epsilon20'  'epsilon70' 'epsilon100')
distance=('max' 'euc' 'cab')

for e in ${epsilon[*]}
do 
    python code/fbb.py -e $e -v 0.9999 -k 10 -a rf -d max
    python code/fbb.py -e $e -v 0.9999 -k 11 -a rf -d max
    python code/fbb.py -e $e -v 0.99 -k 1000 -a none -d euc
done