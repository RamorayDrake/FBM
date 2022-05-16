import torch
from contextlib import contextmanager
import time
from collections import defaultdict
import copy
import pandas as pd
from time import sleep
import os
from nncf.torch.layers import NNCFConv2d
#context manager is the most acurate way to work on CPU, GPU need to be non syncronized
# Edge devices has not been tested yet

models_name = {'resnet50': 'resnet','resnet18': 'resnet','resnet20': 'resnet', 'resnet101': 'resnet','inceptionv3': 'inception',
               'mobilenetv2_w1': 'mobilenet','mobilenet_v2': 'mobilenet','resnet20_cifar10':'resnet'}

def eval_inference(model_name,bit_map=None):
    """ return the latency for model with given bitmap, as exist in the model latency table"""
    model_performance = pd.read_csv('Latency_table_'+model_name+'.csv').to_dict()
    latency = 0.0
    for layer_num in model_performance['Unnamed: 0'].keys():
        name = model_performance['Unnamed: 0'][layer_num]
        if name in bit_map:
            lat_bits = 'latency_' + str(bit_map[name])
            if lat_bits in model_performance:
                latency+= model_performance[lat_bits][layer_num]
            else:
                raise TypeError("Didn't find the bit allocation in latenct table!")
    return latency

def _scale(arch):
    """
    scaling the model latency scheme by model and by bit-width
    """
    model_name = models_name[arch]
    factor_dict = defaultdict(dict)
    #from TVM: https://tvm.apache.org/2019/04/29/opt-cuda-quantized
    #mobilenet: https://pocketflow.github.io/performance/
    factor_dict['8'] = {'resnet': 0.37836,'mobilenet': 0.400,'vgg': 0.31103,'inception': 0.19923,'resnext': 0.12355}
    #from TensortRT: https://developer.nvidia.com/blog/int4-for-ai-inference/
    # and from "Quantization and Deployment of Deep Neural Networks on Microcontrollers" Novac et al

    for k in factor_dict['8']:
        factor_dict['16'][k] = factor_dict['8'][k] * 1.39177
        factor_dict['4'][k]  = factor_dict['8'][k] * 0.85783
        factor_dict['3'][k]  = factor_dict['8'][k] * 0.6338
        factor_dict['2'][k]  = factor_dict['8'][k] * 0.4545
        #factor_dict['1'][k]  = factor_dict['8'][k] * 0.3165
    model_performance = pd.read_csv('Latency_table_'+arch+'.csv')
    model_performance['latency_16'] = model_performance['latency_32']*factor_dict['16'][model_name]
    model_performance['latency_8']  = model_performance['latency_32']*factor_dict['8'][model_name]
    model_performance['latency_4']  = model_performance['latency_32']*factor_dict['4'][model_name]
    model_performance['latency_3']  = model_performance['latency_32'] * factor_dict['2'][model_name]
    model_performance['latency_2']  = model_performance['latency_32']*factor_dict['2'][model_name]
    #model_performance['latency_1']  = model_performance['latency_32']*factor_dict['1'][model_name]
    print(model_performance)
    df = pd.DataFrame(model_performance)
    df.to_csv('Latency_table_'+arch+'.csv',index=False)

def create_model_latency_scheme(model,arch,ds,device, reps=10):
    """
    create model latency table
    model: torch model
    arch: str of the model arch
    ds: dataset used (to create dummy input for inference)
    device: as we measure latency over a given device
    reps: num of reps for warmup the hardware before timing

    return: the dict saved in Latency table
    """
    model_performance = defaultdict(dict)
    ##model
    model = model.to(device)
    if models_name[arch]=='resnet':
        resnet_convs_names = ['conv1','shortcut.0'] #TODO: same for all models supported
        if ds=='imagenet':
            resnet_convs_names = ['conv1', 'downsample.0']
    else: #mobilenet_v2
        resnet_convs_names = ['conv1', 'downsample.0']
    # create input
    img_shape = (1, 3, 32 , 32)
    if ds=='imagenet':
        img_shape = (1, 3, 256, 256)
    x = torch.rand(img_shape, dtype=torch.float).to(device)
    dummy = torch.clone(x)

    #cuda events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # warmup:
        for _ in range(max(reps, 100)):
            model(x)
        initial_conv=True
        for name, m in model.named_modules():
            if isinstance(m,torch.nn.Conv2d) or isinstance(m,NNCFConv2d): #NNCFConv2d
                # save features for the shortcut
                if resnet_convs_names[0] in name:
                    dummy = torch.clone(x)

                start.record()
                if resnet_convs_names[1] in name:
                    for _ in range(reps):
                        m(dummy)
                else:
                    for _ in range(reps):
                        m(x)
                end.record()
                torch.cuda.synchronize()
                time = start.elapsed_time(end)
                if resnet_convs_names[1] not in name:
                    x = m(x)
                if initial_conv:
                    initial_conv = False
                    continue
                model_performance['latency_32'][name] = time / reps


    conv_time=0
    df = pd.DataFrame(model_performance)
    df.to_csv('Latency_table_'+arch+'.csv')#,index=False)
    for layer in model_performance['latency_32'].keys():
        print(f"L: {layer} | TIME: {model_performance['latency_32'][layer]}")
        conv_time+=model_performance['latency_32'][layer]
    print(f'==== inference estimated time:: {conv_time} ====')
    _scale(arch)
    return model_performance



def layers_for_quant_list(model_name):
    "return a list of all quantize conv layers"
    model_performance = pd.read_csv('Latency_table_' + model_name + '.csv').to_dict()
    return list(model_performance['Unnamed: 0'].values())

def assign_bit_allocation(bit_allocation,q_layers_list,quant_selection={},init=None):
    """
    assign the quant_selection for the layers in q_layers_list in bit_config
    init- in not None, use the init value TODO: should implement smart initilization
    """
    if init:
        quant_selection = [init] * len(q_layers_list)
    for i,layer_name in enumerate(q_layers_list):
        bit_allocation[layer_name] =quant_selection[i]
    return bit_allocation

def assign_bit_allocation2(model, bit_allocation):
    act_bits = []
    for name, m in model.named_modules():
        name= name[7:]
        # remove 'pre_ops.0.op' from name
        if name[:-13] in bit_allocation:
            #print(f'name: {name[:-13]}')
            if name.split('.')[-1] == 'op':
                layer_bits = bit_allocation[name[:-13]]
                m.num_bits = layer_bits
                if 'shortcut' not in name and 'downsample' not in name:
                    act_bits.append(layer_bits)
        elif 'external_quantizers.' in name and 'layer' in name and 'relu' in name:  # and name is not 'external_quantizers':
            bits = act_bits.pop(0)
            m.num_bits = bits

def check_model_size(model,bitwidths=None):
    total_bits=0
    if not bitwidths:
        bitwidths = defaultdict(lambda: 32)
    for name, m in model.named_modules():
            bits=sum(param.numel() for param in m.parameters())*bitwidths[name]
            #print(name, bits)
            total_bits+=bits
    print(f'Model size: {total_bits/1e8} MB')
    return total_bits/1e8



##### for CPU only #####
@contextmanager
def timer(device='cpu'):
    start = time.time()
    yield # context breakdown
    end = time.time()
    print(f"This code block executed in {round(end - start, 5)} seconds on {device}.")

#contextmanager class
class CM_tinme:
    def __init__(self):
        self.time = 0
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self,exc_type, exc_obj, exc_tb):
        self.end = time.time()
        self.time = self.end - self.start
        return time
