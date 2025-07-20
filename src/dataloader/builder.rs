use anyhow::{bail, Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::Array2;
use noodles::core::Position;
use noodles::fasta::{
    fai::Index,
    io::{indexed_reader::Builder, IndexedReader},
};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;
use std::{fs::File, io::BufRead, path::Path};

use crate::dataloader::chunk::{DataChunk, DataChunkIter, Sequences, Values};
use crate::dataloader::index::{make_seq_index, ChunkIndex};
use crate::utils::PrefetchIterator;
use crate::w5z::W5Z;

/** Represents a builder for genomic data, allowing for the creation and management of genomic datasets.

    This struct provides methods to create a new genomic dataset, open an existing dataset,
    and manage genomic data chunks. It supports operations like adding files, retrieving chromosome
    information, and iterating over data chunks.

    The builder creates a structured dataset in a specified location:

    root/
    ├── metadata.json
    ├── chr1/
    │   ├── 0/
    │   │   ├── data
    |   |   ├── data.index
    │   │   ├── sequence.dat
    |   |   └── names.txt
    │   |── 1/
    │   │   ├── data
    |   |   ├── data.index
    │   │   ├── sequence.dat
    |   |   └── names.txt

    Parameters
    ----------
    location
        The directory where the genomic data will be stored.
    genome_fasta
        The path to the FASTA file containing the genome sequences.
    segments
        Optional list of genomic segments to include in the dataset.
        The genomic segments should be provided as strings in the format "chrom:start-end".
        If None, the entire genome will be used.
    window_size
        The size of the genomic windows to be processed (default is 524288).
    step_size
        The step size for sliding the window across the genome (default is None, which uses `window_size`).
    chunk_size
        The number of segments to store in each chunk (default is 128).
    resolution
        The resolution of the stored genomic data (default is 1).
    chroms
        A list of chromosomes to include in the dataset. If None, all chromosomes in the FASTA file will be used.
    overwrite
        If True, existing data at the specified location will be overwritten (default is False).

    See Also
    --------
    GenomeDataLoader

    Examples
    --------
    >>> from gdata import as GenomeDataBuilder
    >>> regions = ["chr11:35041782-35238390", "chr11:35200000-35300000"]
    >>> tracks = {'DNase:CD14-positive monocyte': 'ENCSR464ETX.w5z', 'DNase:keratinocyte': 'ENCSR000EPQ.w5z'}
    >>> builder = GenomeDataBuilder("genome", 'genome.fa.gz', segments=regions, window_size=196_608, resolution=128)
    >>> builder.add_files(tracks)
*/
#[pyclass]
pub struct GenomeDataBuilder {
    chrom_sizes: BTreeMap<String, u64>,
    window_size: u64,
    pub(crate) resolution: u64,
    location: PathBuf,
    pub(crate) seq_index: ChunkIndex,
}

impl GenomeDataBuilder {
    pub(super) fn open(location: impl AsRef<Path>) -> Result<Self> {
        location
            .as_ref()
            .join("metadata.json")
            .exists()
            .then_some(())
            .ok_or_else(|| {
                anyhow::anyhow!("No metadata found at {}", location.as_ref().display())
            })?;
        let metadata: Value = serde_json::from_str(&std::fs::read_to_string(
            location.as_ref().join("metadata.json"),
        )?)?;
        let chrom_sizes: BTreeMap<String, u64> = metadata
            .get("chrom_sizes")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("Invalid metadata format"))?
            .iter()
            .map(|(k, v)| {
                let size = v
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("Invalid chromosome size"))?;
                Ok((k.clone(), size))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let index = make_seq_index(location.as_ref(), chrom_sizes.keys());
        let resolution = metadata.get("resolution").unwrap().as_u64().unwrap();
        let window_size = metadata.get("window_size").unwrap().as_u64().unwrap();
        Ok(Self {
            chrom_sizes,
            location: location.as_ref().to_path_buf(),
            window_size,
            resolution,
            seq_index: index,
        })
    }

    fn iter_chunks(&self, trim_target: Option<usize>) -> impl Iterator<Item = DataChunk> {
        let chroms = Box::new(
            self.chrom_sizes
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter(),
        );
        let trim_target = trim_target.map(|t| {
            if t >= self.window_size as usize {
                panic!("Trim target must be less than window size");
            } else if t % self.resolution as usize != 0 {
                panic!(
                    "Trim target must be a multiple of resolution ({})",
                    self.resolution
                );
            } else {
                t / self.resolution as usize
            }
        });
        DataChunkIter {
            root: self.location.clone(),
            chroms,
            chunks: Box::new(std::iter::empty()),
            trim_target,
        }
    }

    /// Iterates over the genomic data chunks, allowing for efficient traversal of genomic datasets.
    pub(super) fn iter_chunk_data(
        &self,
        buffer_size: usize,
        trim_target: Option<usize>,
    ) -> PrefetchIterator<(Sequences, Values)> {
        let iter = self
            .iter_chunks(trim_target)
            .map(|mut chunk| (chunk.get_seqs().unwrap(), chunk.read_all().unwrap()));
        PrefetchIterator::new(iter, buffer_size)
    }
}

#[pymethods]
impl GenomeDataBuilder {
    #[new]
    #[pyo3(
        signature = (
            location, genome_fasta, *, segments=None, window_size=524288, step_size=None, chunk_size=128, resolution=1, chroms=None, overwrite=false,
        ),
        text_signature = "($self, location, genome_fasta, *, segments=None, window_size=524288, step_size=None, chunk_size=128, resolution=1, chroms=None, overwrite=False)"
    )]
    fn new(
        location: PathBuf,
        genome_fasta: PathBuf,
        segments: Option<Vec<String>>,
        window_size: u64,
        step_size: Option<u64>,
        chunk_size: usize,
        resolution: u64,
        chroms: Option<Vec<String>>,
        overwrite: bool,
    ) -> Result<Self> {
        fn write_seqs(
            input: impl IntoIterator<Item = (String, impl IntoIterator<Item = GenomicRange>)>,
            location: &PathBuf,
            fasta_reader: &mut IndexedReader<noodles::fasta::io::BufReader<File>>,
            window_size: u64,
            chunk_size: usize,
        ) -> Result<()> {
            for (chr, segments) in input {
                for (i, chunk) in segments
                    .into_iter()
                    .chunks(chunk_size)
                    .into_iter()
                    .enumerate()
                {
                    let (names, seq): (Vec<_>, Vec<_>) = chunk
                        .map(|segment| {
                            let mut s = get_seq(fasta_reader, &segment)
                                .unwrap()
                                .sequence()
                                .as_ref()
                                .to_vec();
                            let l = segment.len() as u64;
                            if l < window_size {
                                s.resize(window_size as usize, b'N');
                            } else if l > window_size {
                                panic!(
                                    "Segment {} is too long: {} > {}",
                                    segment.pretty_show(),
                                    l,
                                    window_size
                                );
                            }
                            (segment, s)
                        })
                        .unzip();
                    let data = DataChunk::new(location.join(&chr).join(i.to_string()), names)?;
                    data.save_seqs(seq)?;
                }
            }
            Ok(())
        }

        if window_size % resolution != 0 {
            bail!("Window size must be a multiple of resolution");
        }

        if location.exists() {
            if overwrite {
                std::fs::remove_dir_all(&location)?;
            } else {
                bail!("Location already exists: {}", location.display());
            }
        }
        std::fs::create_dir_all(&location)?;

        let mut reader = open_fasta(genome_fasta)?;

        // Retrieve chromosome sizes from the FASTA index
        let mut chrom_sizes: BTreeMap<String, u64> = reader
            .index()
            .as_ref()
            .iter()
            .map(|rec| (rec.name().to_string(), rec.length()))
            .collect();
        if let Some(chroms) = chroms {
            let chroms: HashSet<_> = chroms.into_iter().collect();
            chrom_sizes.retain(|chrom, _| chroms.contains(chrom));
        }

        if let Some(segments) = segments {
            let mut all_chroms = HashSet::new();
            let mut segments = segments
                .into_iter()
                .map(|s| {
                    let mut g = GenomicRange::from_str(&s)
                        .map_err(|_| anyhow::anyhow!("Failed to parse segment '{}'", s))?;
                    all_chroms.insert(g.chrom().to_string());
                    expand_segment(&mut g, window_size, &chrom_sizes);
                    Ok(g)
                })
                .collect::<Result<Vec<_>>>()?;
            segments.sort();
            write_seqs(
                &segments.into_iter().chunk_by(|x| x.chrom().to_string()),
                &location,
                &mut reader,
                window_size,
                chunk_size,
            )?;
            chrom_sizes.retain(|chrom, _| all_chroms.contains(chrom));
        } else {
            let step_size = step_size.unwrap_or(window_size);
            write_seqs(
                get_genome_segments(&chrom_sizes, window_size as u64, step_size),
                &location,
                &mut reader,
                window_size,
                chunk_size,
            )?;
        }

        let metadata = json!({
            "chrom_sizes": chrom_sizes,
            "window_size": window_size,
            "resolution": resolution,
        });
        std::fs::write(
            location.join("metadata.json"),
            serde_json::to_string(&metadata)?,
        )?;

        let seq_index = make_seq_index(&location, chrom_sizes.keys());
        Ok(Self {
            chrom_sizes,
            location,
            window_size,
            resolution,
            seq_index,
        })
    }

    /** Open an existing GenomeDataBuilder instance from a specified location.

       This method checks if the metadata file exists at the given location and initializes
       the GenomeDataBuilder with the stored chromosome sizes, window size, and resolution.

       Parameters
       ----------
       location : Path
           The path to the directory containing the genomic data.

       Returns
       -------
       GenomeDataBuilder
           An instance of GenomeDataBuilder initialized with the existing genomic data.
    */
    #[classmethod]
    #[pyo3(
        name = "open",
        signature = (location),
        text_signature = "(location)",
    )]
    fn open_py(_cls: &Bound<'_, PyType>, location: PathBuf) -> Result<Self> {
        Self::open(location)
    }

    /** Returns a vector of chromosome names present in the genomic data.

       Returns
       -------
       list[str]
           A list of chromosome names as strings.
    */
    fn chroms(&self) -> Vec<&String> {
        self.chrom_sizes.keys().collect()
    }

    /** Returns the keys (track names) in the dataset.

       This method retrieves all keys from the dataset, which are typically the names of files
       containing genomic data. The keys are sorted alphabetically.

       Returns
       -------
       list[str]
           A sorted list of keys as strings.
    */
    pub fn keys(&self) -> Result<Vec<String>> {
        if let Some(chr) = self.chrom_sizes.keys().next() {
            if let Some(entry) = std::fs::read_dir(self.location.join(&chr))?.next() {
                let chunk = DataChunk::open(entry?.path())?;
                Ok(chunk.keys())
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        }
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    fn segments(&self) -> Vec<String> {
        self.seq_index.keys().cloned().collect()
    }

    /** Adds w5z files to the dataset.

       This method processes a batch of files, each associated with a key, and adds them to the genomic data.

       Parameters
       ----------
       files : dict[str, Path]
           A dictionary mapping keys to file paths.
    */
    #[pyo3(
        signature = (files),
        text_signature = "($self, files)",
    )]
    fn add_files(&self, py: Python<'_>, files: IndexMap<String, PathBuf>) -> Result<()> {
        let n_cols = self.window_size / self.resolution;
        files
            .into_iter()
            .chunks(64)
            .into_iter()
            .try_for_each(|chunk| {
                let w5z = chunk
                    .map(|(key, path)| (key, W5Z::open(path).unwrap()))
                    .collect::<Vec<_>>();
                py.check_signals()?;
                self.iter_chunks(None)
                    .chunk_by(|x| x.segments[0].chrom().to_string())
                    .into_iter()
                    .for_each(|(chrom, group)| {
                        let values: Vec<_> = w5z
                            .par_iter()
                            .map(|(k, x)| (k, x.get(&chrom).unwrap().to_vec()))
                            .collect();
                        group.into_iter().for_each(|mut dc| {
                            let segments = dc.segments.clone();
                            let data = values.par_iter().map(|(key, values)| {
                                let arr_elems: Vec<_> = segments
                                    .iter()
                                    .flat_map(|segment| {
                                        let start = segment.start() as usize;
                                        let end = segment.end() as usize;
                                        let mut v: Vec<_> = if self.resolution > 1 {
                                            values[start..end]
                                                .chunks(self.resolution as usize)
                                                .map(|x| {
                                                    let m: average::Mean =
                                                        x.iter().map(|x| *x as f64).collect();
                                                    bf16::from_f64(m.mean())
                                                })
                                                .collect()
                                        } else {
                                            values[start..end]
                                                .iter()
                                                .map(|x| bf16::from_f32(*x))
                                                .collect()
                                        };
                                        if v.len() < n_cols as usize {
                                            v.resize(n_cols as usize, bf16::from_f32(0.0));
                                        }
                                        v
                                    })
                                    .collect();
                                let arr = Array2::from_shape_vec(
                                    (segments.len(), n_cols as usize),
                                    arr_elems,
                                )
                                .unwrap()
                                .into_dyn();
                                (key.to_string(), Values(arr))
                            });
                            dc.save_data(data).unwrap();
                        });
                    });
                Ok(())
            })
    }

    /** Adds a single file to the dataset.

       This method processes a file associated with a key and adds it to the genomic data.
       The data is read from a W5Z file, and stored in chunks. Each chunk has the
       shape (num_segments, num_columns), where num_columns is determined by the
       resolution. The values are averaged if the resolution is greater than 1.

       Parameters
       ----------
       key : str
           The key associated with the file.
       w5z : Path
           The path to the W5Z file containing genomic data.
    */
    #[pyo3(
        signature = (key, w5z),
        text_signature = "($self, key, w5z)",
    )]
    fn add_file(&self, py: Python<'_>, key: &str, w5z: PathBuf) -> Result<()> {
        self.add_files(py, IndexMap::from([(key.to_string(), w5z)]))
    }
}

fn expand_segment(
    segment: &mut GenomicRange,
    window_size: u64,
    chrom_sizes: &BTreeMap<String, u64>,
) {
    if segment.len() < window_size {
        let start = segment
            .start()
            .saturating_sub((window_size - segment.len()) / 2);
        let end = start + window_size;
        segment.set_start(start);
        segment.set_end(end);
    }
    let size = chrom_sizes.get(segment.chrom()).unwrap();
    segment.set_end(segment.end().min(*size));
}

/// Return the segments of the genome as an iterator.
fn get_genome_segments(
    chrom_sizes: &BTreeMap<String, u64>,
    window_size: u64,
    step_size: u64,
) -> impl Iterator<Item = (String, impl Iterator<Item = GenomicRange> + '_)> + '_ {
    chrom_sizes.iter().map(move |(chrom, &size)| {
        let mut start = 0;
        let iter = std::iter::from_fn(move || {
            if start >= size {
                return None;
            }
            let end = (start + window_size).min(size);
            let range = GenomicRange::new(chrom, start, end);
            start += step_size;
            Some(range)
        });
        (chrom.clone(), iter)
    })
}

fn get_seq(
    reader: &mut IndexedReader<noodles::fasta::io::BufReader<File>>,
    region: &impl BEDLike,
) -> Result<noodles::fasta::record::Record> {
    let start = region.start() as usize;
    let end = region.end() as usize;
    if end < start {
        return Err(anyhow::anyhow!(
            "Invalid region: end must be greater than start"
        ));
    }
    let region = noodles::core::region::Region::new(
        region.chrom(),
        Position::try_from(start + 1)?..=Position::try_from(end)?,
    );
    Ok(reader.query(&region)?)
}

fn open_fasta(
    genome_fasta: impl AsRef<Path>,
) -> Result<IndexedReader<noodles::fasta::io::BufReader<File>>> {
    let fai = PathBuf::from(format!("{}.fai", genome_fasta.as_ref().display()));
    let index = if fai.exists() {
        noodles::fasta::fai::io::Reader::new(BufReader::new(File::open(&fai)?))
            .read_index()
            .context("Failed to read FASTA index")?
    } else {
        let reader = noodles::fasta::io::reader::Builder::default()
            .build_from_path(&genome_fasta)
            .with_context(|| {
                format!(
                    "Failed to open FASTA file: {}",
                    genome_fasta.as_ref().display()
                )
            })?;
        create_index(reader.into_inner())?
    };

    let reader = Builder::default()
        .set_index(index)
        .build_from_path(&genome_fasta)?;
    Ok(reader)
}

fn create_index<R: BufRead>(reader: R) -> Result<Index> {
    let mut index = Vec::new();
    let mut reader = noodles::fasta::io::Indexer::new(reader);
    while let Some(record) = reader.index_record()? {
        index.push(record);
    }
    Ok(Index::from(index))
}