#!/bin/bash

f="brfake.txt"

echo "Body,Label" > $f

for i in *
do
	if [[ ("$i" == "mergeNews.sh") || ("$i" == "brfake.txt") ]]
	then
		continue
	else
		(tr '\n\r' ' ' < $i) >> $f
		echo "	1" >> $f
	fi
done

