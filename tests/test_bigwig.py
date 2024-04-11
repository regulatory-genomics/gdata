import gdata
from torch.utils.data import DataLoader

if __name__ == '__main__':
    url = 'https://www.encodeproject.org/files/ENCFF032ZGS/@@download/ENCFF032ZGS.bigWig'
    coords = [('chr1', 1, 1000, False), ('chr2', 10000, 10010, True)]

    data = gdata.PickleData(
        cache='data.p',
        fallback=lambda: gdata.BigWigData(url, coordinates=coords, min_length=1000)
    )
    for x in DataLoader(data, batch_size=1):
        print(x)