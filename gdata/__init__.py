from __future__ import annotations
import torch
from torch.utils.data import IterableDataset
import genomepy
from pathlib import Path
import numpy as np
import gzip
import pickle
import os

class BigWigData(IterableDataset):
    """
    An `IterableDataset` object that yields values from a bigwig file given a list of
    coordinates.

    Parameters
    ----------
    filepath_or_url
        Path/URL to the bigwig file or a list of paths/URLs to multiple bigwig files.
        If multiple files are provided, the values from each file are averaged.
    coordinates 
        A list of tuples with the chromosome name, start and end coordinates,
        and strand (True if foward). Coordinates are 0-based, and half-open.
    min_length
        The returned sequence will be pad with 0.0 if its length is shorter
        than `min_length`.
    
    Returns
    -------
    A tensor of values from the bigwig file.
    """
 
    def __init__(
        self,
        filepath_or_url: str | list[str],
        coordinates: list[tuple[str, int, int, bool]], 
        min_length: int | None = None,
    ):
        self.files = [filepath_or_url] if isinstance(filepath_or_url, str) else filepath_or_url
        self.bigwigs = None
        self.coordinates = coordinates
        self.min_length = min_length

    def __iter__(self):
        import pybigtools
        if self.bigwigs is None:
            self.bigwigs = [pybigtools.open(file) for file in self.files]
        self.coordinates_iter = iter(self.coordinates)
        return self

    def __next__(self):
        chrom, start, end, strand = next(self.coordinates_iter)
        values = None
        for bw in self.bigwigs:
            if values is None:
                values = bw.values(chrom, start, end)
            else:
                values += bw.values(chrom, start, end)
        values /= len(self.bigwigs)
        if not strand:
            values = values[::-1]
        if self.min_length is not None:
            padding_width = self.min_length - len(values)
            if padding_width > 0:
                values = np.pad(values, (0, padding_width), mode='constant', constant_values=(0.0, 0.0))
        return torch.tensor(values, dtype=torch.float32)

class SequenceData(IterableDataset):
    """
    An `IterableDataset` object that yields genomics sequences given a list of
    coordinates or gene ids.

    This class returns a DNA sequence as a tensor of integers (A=0, C=1, G=2, T=3, N=4).
    Note that the sequence is reverse complemented if the strand is negative. 

    Parameters
    ----------
    genome
        Genome object from the `genomepy` package.
    coordinates 
        A list of tuples with the chromosome name, start and end coordinates,
        and strand (True if foward). Coordinates are 0-based, and half-open.
    min_length
        The returned sequence will be pad with 'N' bases if its length is shorter
        than `min_length`.
    
    Returns
    -------
    A DNA sequence as a tensor of integers (A=0, C=1, G=2, T=3, N=4).
    """
    def __init__(
        self,
        genome: genomepy.Genome,
        coordinates: list[tuple[str, int, int, bool]], 
        min_length: int | None = None,
    ):
        super().__init__()
        self.genome = genome
        self.range = None
        self.annotations = None
        self.length = min_length
        self.gene_ids = []
        self.coordinates = coordinates

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
        seq = self.genome.get_seq(chr, start + 1, end, rc=not strand)
        seq = encode_dna(seq, self.length)
        return seq

def encode_dna(dna, length: int | None = None, padding: str = 'right') -> torch.LongTensor:
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

class PickleData(IterableDataset):
    """
    Deserialize dataset from the cache file. When the file is not present, data
    are generated using the provided fallback function.

    This class is useful when you want to cache the generated data to avoid
    recomputing it every time you run the script, especially for heavy computations.

    Parameters
    ----------
    cache
        File path to the cache file.
    fallback
        A function that generates the data when the cache file is not present.
    chunk_size
        Number of items to store in each chunk of the cache file.
    """
    def __init__(
        self,
        cache: str,
        fallback=None,
        chunk_size: int = 200,
    ):
        super().__init__()
        self.cache = cache
        self.fallback = fallback
        self.chunk_size = chunk_size
        self.iterator = None

    def __iter__(self):
        if not Path(self.cache).exists():
            print(f'Cache "{self.cache}" does not exist, fall back to the default generation function')
            if self.fallback is None:
                raise ValueError('No fallback function provided')
            _save_data(self.fallback(), self.cache, self.chunk_size)
        else:
            print(f'Generating dataset from the cache "{self.cache}"')
        self.iterator = _FromPickle(self.cache).__iter__()
        return self

    def __next__(self):
        return self.iterator.__next__()

class _FromPickle(IterableDataset):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__()
        self.filename = filename
        self.file = None
        self.chunk = None

    def __iter__(self):
        if self.file is not None:
            self.file.close()
        self.file = gzip.open(self.filename, 'rb')
        return self

    def __next__(self):
        if self.chunk is None:
            try:
                chunk = pickle.load(self.file)
                self.chunk = iter(chunk)
            except EOFError:
                self.file.close()
                raise StopIteration
        
        try:
            return next(self.chunk)
        except StopIteration:
            self.chunk = None
            return self.__next__()

def _save_data(data, filename, chunk_size: int = 200):
    try:
        with gzip.open(filename, 'wb') as file:
            chunk = []
            for item in data: 
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    pickle.dump(chunk, file)
                    chunk = []
            if len(chunk) > 0:
                pickle.dump(chunk, file)
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        raise e