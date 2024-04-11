import gdata
from torch.utils.data import DataLoader

if __name__ == '__main__':
    url1 = 'https://www.encodeproject.org/files/ENCFF822HBC/@@download/ENCFF822HBC.bigWig'
    url2 = 'https://www.encodeproject.org/files/ENCFF673SGE/@@download/ENCFF673SGE.bigWig'
    coords = [('chr1', 1, 1000, False), ('chr2', 10000, 10010, True)]

    data = gdata.PickleData(
        cache='data.p',
        fallback=lambda: gdata.BigWigData([url1, url2], coordinates=coords, min_length=1000)
    )
    for x in DataLoader(data, batch_size=1):
        print(x)