import gdata
import genomepy
from torch.utils.data import DataLoader

if __name__ == '__main__':
    genome = genomepy.Genome('GRCh38')
    print(genome)

    field = 'gene_type'
    annotation = genomepy.Annotation(genome.genome_dir)
    gtf = annotation.gtf
    #print(gtf)

    coords = [('chr1', 1, 1000, False)]
    gene_ids = ["ENSG00000278625.1"]

    data = gdata.SequenceData(genome, gene_ids=gene_ids)
    for x in DataLoader(data, batch_size=1):
        print(x)