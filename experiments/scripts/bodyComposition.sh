#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=ZF

#array=( $@ )
#len=${#array[@]}
#EXTRA_ARGS=${array[@]:3:$len}
#EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
TRAIN_IMDB="bodyComposition_2016_trainval"
TEST_IMDB="bodyComposition_2016_test"
PT_DIR="bodyComposition"
ITERS=$1


LOG="experiments/logs/bodyComposition_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/bodyComposition.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
#
#time ./tools/test_net.py --gpu ${GPU_ID} \
#  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
#  --net ${NET_FINAL} \
#  --imdb ${TEST_IMDB} \
#  --cfg experiments/cfgs/bodyComposition.yml \
#  --vis \
#  --threshold 0.2
#  ${EXTRA_ARGS}