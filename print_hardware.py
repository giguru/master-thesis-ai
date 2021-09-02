import torch, platform, psutil, cpuinfo, sys, pip
use_cuda = torch.cuda.is_available()


__all__ = ['print_hardware']


def header(string):
    print('------', string, '------')


def print_hardware(save_to_disk: bool = True):
    platform_uname = platform.uname()

    header('Hardware info')
    print('Platform system :', platform.system())
    print('Platform version:', platform_uname.version)
    cpu = cpuinfo.get_cpu_info()
    print('CPU             :', cpu['brand_raw'])
    print('RAM             :', str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB")

    header('CUDA information')
    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        n_devices = torch.cuda.device_count()
        print('__Number CUDA Devices:', n_devices)
        print('__CUDA Device Name:', [torch.cuda.get_device_name(i) for i in range(n_devices)])
        print('__CUDA Device Memory [GB]:', [torch.cuda.get_device_properties(0).total_memory / 1e9 for i in range(n_devices)])
    else:
        print('Not present in system')

    header('Software info')
    print('Python version  :', platform.python_version())
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze
    packages = [p for p in freeze.freeze()]
    print('Packages        :', packages[0])
    for p in packages[1:]:
        print('                 ', p)



print_hardware()

