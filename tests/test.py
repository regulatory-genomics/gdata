import gdata
import genomepy
from torch.utils.data import DataLoader

if __name__ == '__main__':
    genome = genomepy.Genome('GRCh38')

    coords = [('chr1', 1, 1000, False), ('chr21', 10000, 10010, True)]
    gene_ids = ["ENSG00000278625.1"]

    data = gdata.PickleData(
        filename='data.p',
        fallback=lambda: gdata.SequenceData(genome, coordinates=coords)
    )
    for x in DataLoader(data, batch_size=1):
        print(x)