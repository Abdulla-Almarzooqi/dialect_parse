#!/bin/sh

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH --exclude=cn008


export ROOT=$(dirname "$SCRIPT")
export DATA=./data
export TRAIN=$DATA/patb-lev_train.conllx
export DEV=$DATA/Levantine/lev_dev.conllx
export TEST=$DATA/Levantine/lev_test.conllx
export CONFIG=$ROOT/biaffine.bert.default.ini
export BERT=ashabrawy/lev_aligned_msa
export MODEL=models/aligned_lev.model

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
		--proj > lev_out.log
