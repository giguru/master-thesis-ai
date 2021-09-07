import torch, platform, psutil, cpuinfo
use_cuda = torch.cuda.is_available()


__all__ = ['print_hardware']


def header(string):
    print('------', string, '------')


def print_hardware(save_to_disk: bool = True):
    platform_uname = platform.uname()
    cpu = cpuinfo.get_cpu_info()
    gb_string = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"

    header('Hardware info')
    print('Platform system :', platform.system())
    print('Platform version:', platform_uname.version)
    print('CPU             :', cpu['brand_raw'])
    print('RAM             :', gb_string)

    header('CUDA information')
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        total_gpu_memory = [torch.cuda.get_device_properties(0).total_memory / 1e9 for i in range(n_devices)]
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', n_devices)
        print('__CUDA Device Name:', [torch.cuda.get_device_name(i) for i in range(n_devices)])
        print('__CUDA Device Memory [GB]:', total_gpu_memory)
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

    header('For in your publication')
    print(f"Experiments were done on a {cpu['hz_advertised_friendly']} CPU, {gb_string} RAM system"\
          +(f"with {n_devices} GPUs with {total_gpu_memory} VRAM in total." if torch.cuda.is_available() else " without GPUs")
          +"."
          )
