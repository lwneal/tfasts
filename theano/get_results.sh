#!/bin/sh

if [ $# -lt 1 ]; then
    echo "Usage: $0 TAG"
    echo "TAG: any mnemonic tag for this run"
    exit
fi

IP=52.88.104.238
TAG=$1
NAME=`date "+%Y_%m_%d"`_$TAG

for DIR in output demos comparisons; do
    mkdir ${DIR}_${NAME}
    scp -r ${IP}:ml-demo/${DIR}/* ${DIR}_${NAME}
done

open comparisons_${NAME}
python eval.py --equal-precision-recall output_$NAME
