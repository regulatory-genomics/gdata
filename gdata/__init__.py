from __future__ import annotations
import torch
import torchdata
from torch.utils.data import IterableDataset
import genomepy

class SequenceData(IterableDataset):
    """
    An `IterableDataset` object that yields consecutive genomic sequences.

    This dataset reads DNA sequences from a FASTA file and returns them as sub-sequences of fixed length.

    Parameters
    ----------
    genome
        Genome object from the `genomepy` package.
    coordinates 
        Coordinates are 1-based.
    gene_ids
        A list of gene ids to extract sequences from. The strand of the gene is
        used to determine the orientation of the sequence.
    upstream
        Number of nucleotides to include upstream of the gene.
    downstream
        Number of nucleotides to include downstream of the gene.
    use_raw_seq
        If True, return the raw DNA sequence instead of encoding it
    data_only
        If True, return only the DNA sequences, otherwise return a tuple of ((name, start, end), sequence)
    """
    def __init__(
        self,
        genome: genomepy.Genome,
        *,
        coordinates: list[tuple(str, int, int, bool)] | None = None,
        gene_ids: list[str] | None = None,
        upstream: int = 2000,
        downstream: int = 2000,
        use_raw_seq: bool = False,
        data_only: bool = True,
    ):
        super().__init__()
        self.genome = genome
        self.upstream = upstream
        self.downstream = downstream
        self.use_raw_seq = use_raw_seq
        self.data_only = data_only
        self.range = None
        self.annotations = None
        if coordinates is not None:
            self.coordinates = coordinates
        elif gene_ids is not None:
            self.coordinates = [self._find_location(x) for x in gene_ids]
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
        chr, start, end, rc = self.coordinates[i]
        seq = self.genome.get_seq(chr, start, end, rc)
        if not self.use_raw_seq:
            seq = encode_dna(seq)
        if self.data_only:
            return seq
        else:
            return ((chr, start, end), seq)

    def _find_location(self, name: str):
        if self.annotations is None:
            self.annotations = _read_annotation(self.genome, 'gene_id')

        if name in self.annotations.index:
            rec = self.annotations.loc[name]
            chr = rec['seqname']
            rc = True if rec['end'] == '-' else False
            start = rec['start']
            end = rec['end']
            if rc:
                start = start - self.downstream
                end = end + self.upstream
            else:
                start = start - self.upstream
                end = end + self.downstream
            return (chr, start, end, rc)
        else:
            raise ValueError(f'Gene {name} not found in genome annotation')

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