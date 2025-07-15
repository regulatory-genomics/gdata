import numpy as np
import pandas as pd
from pathlib import Path
from gdata._gdata import W5Z, bam_cov, bw_to_w5z

__all__ = ['bam_cov', 'balanced_sampling', 'bw_to_w5z', 'W5Z', 'GenomeData']

def balanced_sampling(data, num_bins, samples_per_bin, return_indices=False):
    df = pd.DataFrame(data, columns=['value'])
    df['index_original'] = df.index
    df['bin'] = pd.cut(df['value'], bins=num_bins, labels=range(num_bins))

    key = 'index_original' if return_indices else 'value'
    # Perform sampling
    # Get indices of sampled data
    samples = df.groupby('bin', observed=True).apply(
        lambda x: x.sample(min(len(x), samples_per_bin))[key] if len(x) > 0 else x
    ).reset_index(drop=True)
    
    return samples