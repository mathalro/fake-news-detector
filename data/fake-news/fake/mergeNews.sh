#!/bin/bash

echo "Body,Label" > brfake.txt

for i in *
do
	if [[ ("$i" == "mergeNews.sh") || ("$i" == "brfake.txt") ]]
	then
		continue
	else
		(tr '\n\r' ' ' < $i) >> brfake.txt
		echo >> brfake.txt
	fi
done
