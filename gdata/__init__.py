from __future__ import annotations
import torch
from torch.utils.data import IterableDataset
import genomepy

class SequenceData(IterableDataset):
    """
    An `IterableDataset` object that yields genomics sequences given a list of
    coordinates or gene ids.

    This class returns a DNA sequence as a tensor of integers (A=0, C=1, G=2, T=3, N=4)
    or a tuple of ((name, start, end), sequence) if `data_only` is False.
    Note that the sequence is reverse complemented if the strand is negative. 
    When `gene_ids` is provided, the coordinates are extracted from the genome annotation.
    Not all gene ids may be found in the annotation file, in which case the sequence is skipped.
    The `.gene_ids` attribute contains the list of gene ids that were found.

    Parameters
    ----------
    genome
        Genome object from the `genomepy` package.
    coordinates 
        A list of tuples with the chromosome name, start and end coordinates,
        and strand (True if foward). Coordinates are 1-based.
    gene_ids
        A list of gene ids to extract sequences from. 
    upstream
        Number of nucleotides to include upstream of the gene.
    downstream
        Number of nucleotides to include downstream of the gene.
    data_only
        If True, return only the DNA sequences, otherwise return a tuple of ((name, start, end), sequence)
    ignore_ensembl_suffix
        If True, ignore the suffix in ENSEMBL gene ids.
    
    Returns
    -------
    A DNA sequence as a tensor of integers (A=0, C=1, G=2, T=3, N=4) or a tuple of
    ((name, start, end), sequence) if `data_only` is False. 
    """
    def __init__(
        self,
        genome: genomepy.Genome,
        *,
        coordinates: list[tuple(str, int, int, bool)] | None = None,
        gene_ids: list[str] | None = None,
        upstream: int = 2000,
        downstream: int = 2000,
        data_only: bool = True,
        ignore_ensembl_suffix: bool = True,
    ):
        super().__init__()
        self.genome = genome
        self.upstream = upstream
        self.downstream = downstream
        self.data_only = data_only
        self.ignore_enesmbl_suffix = ignore_ensembl_suffix
        self.range = None
        self.annotations = None
        self.length = None
        self.gene_ids = []
        if coordinates is not None:
            self.coordinates = coordinates
        elif gene_ids is not None:
            self.length = upstream + downstream
            coordinates = []
            for id in gene_ids:
                loc = self._find_location(id)
                if loc is not None:
                    coordinates.append(loc)
                    self.gene_ids.append(id)
            self.coordinates = coordinates
        else:
            raise ValueError('Either coordinates or gene_ids must be provided')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.range = iter(range(len(self.coordinates)))
        else:  # in a worker process
            raise NameError("not implemented")
            #per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            #worker_id = worker_info.id
            #iter_start = self.start + worker_id * per_worker
            #iter_end = min(iter_start + per_worker, self.end)
            #return iter(range(iter_start, iter_end))
        return self

    def __next__(self):
        i = next(self.range)
        chr, start, end, strand = self.coordinates[i]
        seq = self.genome.get_seq(chr, start, end, rc=not strand)
        seq = encode_dna(seq, self.length)
        if self.data_only:
            return seq
        else:
            return ((chr, start, end), seq)

    def _find_location(self, name: str):
        if self.ignore_enesmbl_suffix:
            name = name.split('.')[0] if name.startswith('ENS') else name

        if self.annotations is None:
            gtf = _read_annotation(self.genome, 'gene_id')
            if self.ignore_enesmbl_suffix:
                gtf.index = [x.split('.')[0] if x.startswith('ENS') else x for x in gtf.index]
            self.annotations = gtf

        if name in self.annotations.index:
            rec = self.annotations.loc[name]
            chr = rec['seqname']
            strand = True if rec['end'] == '+' else False
            start = rec['start']
            end = rec['end']
            if strand:
                start = start - self.downstream
                end = end + self.upstream
            else:
                start = start - self.upstream
                end = end + self.downstream
            return (chr, start, end, strand)
        else:
            return None

def _read_annotation(genome, key: str, feature: str ='gene'):
    annotation = genomepy.Annotation(genome.genome_dir)
    gtf = annotation.gtf
    gtf = gtf[gtf['feature'] == feature]
    gtf.index = annotation.from_attributes(key, annot=gtf)
    return gtf

def encode_dna(dna, length: int | None = None) -> torch.LongTensor:
    """
    Convert DNA strings to integers.
    """
    codes = [_nucle_to_int(x) for x in dna]
    if length is not None and len(codes) < length:
        codes += [4] * (length - len(codes))
    return torch.tensor(codes)

def _nucle_to_int(x: str) -> int:
    if x == 'A' or x == 'a':
        return 0
    elif x == 'C' or x == 'c':
        return 1
    elif x == 'G' or x == 'g':
        return 2
    elif x == 'T' or x == 't':
        return 3
    elif x == 'N' or x == 'n':
        return 4
    else:
        raise ValueError(f'Unknown nucleotide {x}')

def _next_token(
    inputs: torch.LongTensor
)  -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Prepare inputs/labels for next token prediction.
    """
    return (inputs[:-1], inputs[1:])

def _mask_nucleotides(
    inputs: torch.LongTensor,
    mask_prob: float = 0.15,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Prepare masked inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original.

    Parameters
    ----------
    inputs
        Input DNA sequence
    mask_prob
        Probability of masking a nucleotide

    Returns
    -------
    A tuple of masked input sequence and labels (with -100 for non-masked nucleotides)
    """

    labels = inputs.clone()

    # We sample a few nucleotides in each sequence for MLM training (with probability `mask_prob`)
    probability_matrix = torch.full(labels.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input bases with N (id: 4)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = 4

    # 10% of the time, we replace masked input nucleotides with random nucleotides
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(5, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels