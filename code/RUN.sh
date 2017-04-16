#!/bin/bash
data=$1
results=$2
FOLDS=$3

mkdir $results

METHODS0=("ContextLMJoint")
for ((FOLDTORUN=0;FOLDTORUN<FOLDS;FOLDTORUN+=1));
do
	for METHOD in ${METHODS0[*]}
	do
		echo "Running fold "$FOLDTORUN" METHOD "$METHOD" with training data range 0"
		python main.py $FOLDTORUN $METHOD 0 $data $results > $results/$FOLDTORUN"_"$METHOD"_0"
	done
done

#analyze results
python gather_results.py $results micro
