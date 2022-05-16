class ALU():
    def __init__(self,n: int, time, bits=8,split=False):
        """
        Arithmetic logic unit that perform calculations
        :param n: num of ALUs
        :param time:
        :param bits:
        todo: implement split, can perform two 4bit ops in one 4bit unit?
        """
        self.n = n
        self.time = time
        self.bits = bits
        self.occupy = 0
    def __call__(self,input_size):
        assert input_size>=0


        if input_size % (self.n*self.bits):
            return ((input_size // (self.n*self.bits)) + 1) * self.time
        return (input_size // (self.n * self.bits)) * self.time
    def __str__(self):
        return f'CPU has {self.n} ALUs of {self.bits} bits and takes {self.time} msec to perform'

class memory_unit():
    def __init__(self,size,time,clear_time=0):
        """
        class to describe memory unit (l1 cache, ram)
        :param size: size of memory unit [page size]
        :param time: time to fetch from memory [msec]
        :param clear_time: time to clear memory back to disk [msec]
        (don't know if needed, if not might need to make sure to use the time of next level memory when clearing)
        todo: add bus size that change time?
        """
        assert size > 0
        assert time >= 0.

        self.size=size
        self.time=time
        self.clear_time = clear_time
        self.occupy = 0
    def __call__(self,input_size,clear=False):
        """
        saving input_size in memory
        :param input_size: input size needed [Bit]
        :param clear: flag if need to clear memory before the forward (not sure that need)
        :return:time and reminder memory consumption needed
        """
        assert input_size>=0
        time=0
        if clear:
            self.occupy=0
            time+=self.clear_time
        if input_size + self.occupy  > self.size:
            self.occupy = self.size
            rem = input_size %  self.size
        else:
            self.occupy = input_size + self.occupy
            rem = 0
        return time + self.time, rem
        #return self.size * self.time, rem
    def __str__(self):
        return f'Memory has {self.size} Bits and takes {self.time} msec to fetch'

class HW_arch():
    def __init__(self, Cpu :ALU,Memory: list):
        self.Cpu = Cpu
        self.Memory = Memory
    def __call__(self,input_size):
        pass


class LatnecyLoss(nn.Module):
    def __init__(self,comp,L1,L2, channels, input_size=56):
        """
        Latnecy Loss
        should
        :param channels:
        :param steps:
        :param strides:
        :param input_size:
        """
        super(LatnecyLoss, self).__init__()

        self.channels = channels

