import gdata
from torch.utils.data import DataLoader

if __name__ == '__main__':
    file = '/home/zhangkaiLab/zhangkai33/lab_home/shared/data/public_HiC/K562/ENCFF621AIY.hic'
    coords = [('chr1', 10000000, 10000500), ('chr2', 10000000, 10000500)]

    data = gdata.HiCData(file, coordinates1=coords, resolution=100, format='matrix')
    for x in data:
        print(x)