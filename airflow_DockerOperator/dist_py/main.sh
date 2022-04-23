#!/bin/bash


# train or inference
process_1=$1

# preprocess, create_graph, train/infer
process_2=$2


if [ $process_1 = 'train' ]
then 
    config='./config_files/config_train.yml'

elif [ $process_1 = 'inference' ]
then
    config='./config_files/config_inf.yml'

else
    echo "you should only choose train or inference"
fi


if [ $process_2 = 'preprocess' ]
then
    file='preprocessing.py'

elif [ $process_2 = 'create_graph' ]
then
    file='create_graph.py'

elif [ $process_2 = 'train' ]
then
    file='training.py'

elif [ $process_2 = 'inference' ]
then
    file='inference.py'

else
    echo "you should choose preprocess, create_graph, train, inferece"
fi


python $file --config-path $config