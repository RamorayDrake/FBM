

# SAQnn: Spec Aware Quantization for neural networks

This work uses Intel Neural Network Compression Framework [NNCF](https://github.com/openvinotoolkit/nncf#user-content-installation)

We use the [Fisher information](https://arxiv.org/pdf/1705.01064.pdf) (the Empirical Fisher) to estimate the hessian trace and have smart initilization for the probabilities of quantizations.


## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* **To install** and develop locally:
```bash
git clone https://github.com/mkimhi/saqnn.git
cd saqnn
pip install -r requirements.txt
```

## Getting Started
### Cifar10
```
python3 quant_train.py -a resnet18 --pretrained --epochs 50 --lr 0.001 -b 512 --ds cifar10 --data ./ --save-path checkpoints/ --wd 1e-4 -p 50 -qf 1
```
### Imagenet
```
python3 quant_train.py -a resnet18 --pretrained --epochs 50 --lr 0.001 -b 128 --ds Imagenet --data PATH_TO_IMAGENET --save-path checkpoints/ --wd 1e-4 -p 50 -qf 1
```
### Fine tune
```
python3 quant_train.py -a resnet18 --epochs 50 --lr 0.0001 -b 512 --ds cifar10 --data ./ --save-path checkpoints/ --wd 1e-5 -p 50 --resume checkpoints/RUN_PATH/model_best.pth.tar --resume-quant --distill-method KD_naive --teacher-arch resnet18_cifar10
```


## TODO:
1. fix imagenet second epoch drop **Moshe and Tal**
2. finish Imagenet for resnet18 and resnet50 **Moshe**
3. Fine tune best models with self-KD **Tal**
4. run mobile-net V2 **Moshe** 
5. finish read related work, write realted work **Moshe** 
6. find all latency-accuracy tradeoffs of related work, put in tables and make a graph (latenct-acc graph) **Moshe**
8. Write the method- explain the entropy derivitive **Moshe**
9. Write intro **Moshe**
11. add abstruct and conclusion **ALL**
12. accepted to NeuroIPS **ALL**

## Nice to have:
1. Use timm models as baseline
2. extract loss graphs from Tensorboard **Tal**
3. cifar graph with $\alpha$ and $\beta$ as axis and ACC/LAT as value

## Future work
1. use LSQ as quant beckend
2. add power estimation and acc-lat-pow tradeoffs
3. add binarization/Trinary/both to mixed-precision 


## License
released under the [MIT license](LICENSE).
