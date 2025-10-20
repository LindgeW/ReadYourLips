import numpy as np

# 同一文件追加内容
def write_to(fn, data):
    with open(fn, 'a') as f:
        f.write(data)
        f.write('\n')


def write_numpy_to(fn, arr):
    assert fn.endswith('.txt')
    with open(fn, 'a') as f:
        np.savetxt(f, arr)
