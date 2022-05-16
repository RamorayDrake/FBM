

# FBM: Fast-Bit Allocation for Mixed-Precision
Quantization

This work uses Intel Neural Network Compression Framework [NNCF](https://github.com/openvinotoolkit/nncf#user-content-installation)

We use the [Fisher information](https://arxiv.org/pdf/1705.01064.pdf) (the Empirical Fisher) to estimate the hessian trace and have smart initilization for the probabilities of quantizations.


## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* **To install** and develop locally:
```bash
git clone https://github.com/RamorayDrake/FBM.git
cd FBM
pip install -r requirements.txt
```

## Getting Started
### Cifar10
```
python3 quant_train.py -a resnet18 --pretrained --epochs 50 --lr 0.001 -b 512 --ds cifar10 --data ./ --save-path checkpoints/ --wd 1e-4 -p 50 -qf 1 --create_table
```
### Imagenet
```
python3 quant_train.py -a resnet18 --pretrained --epochs 50 --lr 0.001 -b 128 --ds Imagenet --data PATH_TO_IMAGENET --save-path checkpoints/ --wd 1e-4 -p 50 -qf 1 --create_table
```
### Fine tune
```
python3 quant_train.py -a resnet18 --epochs 50 --lr 0.0001 -b 512 --ds cifar10 --data ./ --save-path checkpoints/ --wd 1e-5 -p 50 --resume checkpoints/RUN_PATH/model_best.pth.tar --resume-quant --distill-method KD_naive --teacher-arch resnet18_cifar10
```


## Updated:

## Future work
1. add power estimation to all benchmarks
2. add hardware simulator results from our partners 
3. add Trinary to mixed-precision quantization 
4. add CO2 consumption of our training procedure


## License
released under the [MIT license](LICENSE).
