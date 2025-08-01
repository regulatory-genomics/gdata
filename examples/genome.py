from gdata import GenomeDataBuilder, GenomeDataLoader
from pathlib import Path

if not Path("test_genome").exists():
    builder = GenomeDataBuilder(
        "test_genome",
        '/data/Public/genome/GRCh38/GRCh38.primary_assembly.genome.fa.gz',
        segments=["chr11:35041782-35238390", "chr11:35200000-35300000"],
        window_size=196_608,
        resolution=1,
    )
    builder.add_files(
        {
            'ChIP-H3K27ac:keratinocyte': 'ENCSR666TFS.w5z',
            'DNase:CD14-positive monocyte': 'ENCSR464ETX.w5z',
            'DNase:keratinocyte': 'ENCSR000EPQ.w5z',
            'dorsolateral_prefrontal_cortex-CTCF': 'dorsolateral_prefrontal_cortex-CTCF.w5z',
        }
    )

loader = GenomeDataLoader(
    "test_genome",
    #trim_target=40_960,
    trim_target=40_960*2,
    resolution=1,
    seq_as_string=True,
)
print(loader)

loader.plot(
    'chr11:35041782-35238390',
    loader.tracks,
    savefig="test_genome_signal.png",
)
