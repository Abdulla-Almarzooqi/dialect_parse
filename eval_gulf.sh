#!/bin/sh

#SBATCH -n 20

export ROOT=$(dirname "$SCRIPT")
export CONFIG=$ROOT/biaffine.bert.default.ini
export MODEL=models/aligned_gulf.model 
export DATA=data/Gulf/gumar_test.conllu


python parser/supar/cmds/dep/biaffine.py evaluate \
 --device 0 \
 --conf $CONFIG \
 --path $MODEL \
 --data $DATA \
 --tree \
 --proj > best_model_eval_gulf.log
