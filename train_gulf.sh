#!/bin/sh

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH --exclude=cn008


export ROOT=$(dirname "$SCRIPT")
export DATA=./data
export TRAIN=$DATA/patb-gumar_train.conllx
export DEV=$DATA/Gulf/gumar_dev.conllu
export TEST=$DATA/Gulf/gumar_test.conllu
export CONFIG=$ROOT/biaffine.bert.default.ini
export BERT=ashabrawy/gumar_aligned_msa
export MODEL=models/aligned_gulf.model

python parser/supar/cmds/dep/biaffine.py train \
                --device 0 \
                --conf $CONFIG \
                --path $MODEL \
                --train $TRAIN \
                --dev $DEV \
                --test $TEST \
                --encoder=bert \
               	--bert=$BERT \
	       	--tree \
		--proj	> gulf_out.log
