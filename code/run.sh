#!/bin/bash

variance=(0.99999)
kvals=(3000)
model=('none' 'rf' 'rbfSVM')
epsilon=('baseline' 'epsilon20' 'epsilon40' 'epsilon50')
distance=('max' 'euc' 'cab')

for e in ${epsilon[*]}
do
    for m in ${model[*]}
    do
        for d in ${distance[*]}
        do
            python code/fbb.py -e $e -v $variance -k $kvals -a $m -d $d
        done
    done
done