#!/bin/bash

f="brtrue.txt"

echo "Body,Label" > $f

for i in *
do
	if [[ ("$i" == "mergeNews.sh") || ("$i" == $f) ]]
	then
		continue
	else
		(tr '\n\r' ' ' < $i) >> $f
		echo "	0" >> $f
	fi
done

