#!/bin/bash

text='default norm stem'
embed='selftrain_100 selftrain_300 fasttext'
arch='cnn capsule'

echo ""
echo "*************************************************************"
echo Test all Neural network models
for a in $arch
do
    for t in $text
    do
        for e in $embed
        do
            echo ""
            echo Test Neural network with $a, features are $t csv file and $e embedding
            python keras_nlp.py --mode test --architecture $a --text $t --embed $e
            echo ""
            echo "*************************************************************"
        done
    done
done
