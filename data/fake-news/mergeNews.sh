#!/bin/bash

echo "Body,Label" > brfake.txt

for i in *
do
	if [[ ("$i" == "mergeNews.sh") || ("$i" == "brfake.txt") ]]
	then
		continue
	else
		echo -n "\"" >> brfake.txt
		cat $i >> brfake.txt
		echo "\",1" >> brfake.txt
	fi
done
