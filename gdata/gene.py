from __future__ import annotations
import genomepy
from collections.abc import Sequence

def get_promoters(
    genome: genomepy.Genome,
    gene_ids: Sequence[str],
    upstream: int = 2000,
    downstream: int = 2000,
    gene_type=None,
    ignore_ensembl_suffix=False,
) -> list[tuple[str, int, int, bool]]:
    """
    Return promoter regions of given gene list.

    Parameters
    ----------
    genome
        Genome object from the `genomepy` package.
    gene_ids
        A list of gene ids to extract sequences from. 
    upstream
        Number of nucleotides to include upstream of the gene.
    downstream
        Number of nucleotides to include downstream of the gene.
    gene_type
        Filter genes by gene type (e.g., protein_coding).
    ignore_ensembl_suffix
        If True, ignore the suffix in ENSEMBL gene ids.
    
    Returns
    -------
    Promoters represented as a list of tuples with the chromosome name, start and end coordinates, strand.
    """
    gtf = _read_annotation(genome, 'gene_id', gene_type=gene_type)
    if ignore_ensembl_suffix:
        gtf.index = [x.split('.')[0] if x.startswith('ENS') else x for x in gtf.index]
    
    result = []
    for gene in gene_ids:
        name = gene
        if ignore_ensembl_suffix:
            name = name.split('.')[0] if name.startswith('ENS') else name
        if name in gtf.index:
            rec = gtf.loc[name]
            chr = rec['seqname']
            strand = rec['strand']
            if strand == '+':
                strand = True
            elif strand == '-':
                strand = False
            else:
                raise ValueError(f'Unknown strand {strand}')
            start = rec['start'] - 1  # Convert 1-based coordinate to 0-based coordinate
            end = rec['end'] - 1
            if strand:
                tss = start
                start = max(tss - upstream, 0)
                end = tss + downstream
            else:
                tss = end
                start = max(tss - downstream + 1, 0)
                end = tss + upstream + 1
            loc = (chr, start, end, strand)
        else:
            loc = None
        result.append((gene, loc))
    return result

def _read_annotation(genome, key: str, feature: str ='gene', gene_type: str | None = None):
    annotation = genomepy.Annotation(genome.genome_dir)
    gtf = annotation.gtf
    gtf = gtf[gtf['feature'] == feature]
    gtf.index = annotation.from_attributes(key, annot=gtf)

    if gene_type is not None:
        gene_type = annotation.from_attributes('gene_type', annot=gtf)
        gtf = gtf[gene_type == gene_type]

    return gtf

