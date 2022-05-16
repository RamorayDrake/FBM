#!/bin/bash

lr=0.001
data_set=cifar10
model=resnet18
epochs=50
batch_size=512
data=../talco/comp/
path=checkpoints/
wd=1e-4
print=50
quant_freq=1


for t in 0.1 0.5 1 3
do
  for a in 0.1 1 5 10 20 50 70 100 500
  do
    for b in 0.1 1 5 10 20 50 70 100 500
    do
      python3 ./quant_train.py \
              -T $t \
              --alpha $a \
              --beta $b \
              -a ${model} \
              --lr ${lr} \
              --pretrained\
              --epochs ${epochs} \
              --ds ${data_set} \
              -b ${batch_size} \
              --data ${data} \
              --save-path ${path} \
              --wd ${wd} \
              -p ${print} \
              -qf ${quant_freq}
    done
  done
done
