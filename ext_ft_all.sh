#!/bin/bash

text='default norm stem'
embed='selftrain_100 selftrain_300 fasttext'

echo ""
echo "*************************************************************"
echo Extract numeric and TFIDF
for t in $text
do
    echo ""
	echo Extract $t csv file
	python extract_features.py --text $t
	echo "*************************************************************"
    echo ""
done


echo ""
echo "*************************************************************"
echo Extract word feature
for t in $text
do
    for e in $embed
    do
        echo ""
        echo Extract $t csv file, $e embedding
        python extract_word.py --text $t --embed $e
        echo ""
        echo "*************************************************************"
    done
done
