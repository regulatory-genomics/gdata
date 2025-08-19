use anyhow::{Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use indexmap::IndexMap;
use itertools::Itertools;
use noodles::fasta::{
    fai::Index,
    io::{indexed_reader::Builder, IndexedReader},
};
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;
use std::{fs::File, io::BufRead, path::Path};
use tempfile::TempDir;

use super::data_store::DataStoreBuilder;
use crate::w5z::W5Z;

/** Represents a builder for genomic data, allowing for the creation and management of genomic datasets.

    This struct provides methods to create a new genomic dataset, open an existing dataset,
    and manage genomic data chunks. It supports operations like adding files, retrieving chromosome
    information, and iterating over data chunks.

    Parameters
    ----------
    location
        The directory where the genomic data will be stored.
    genome_fasta
        The path to the FASTA file containing the genome sequences.
    window_size
        The size of the genomic windows to be processed.
    segments
        Optional list of genomic segments to include in the dataset.
        The genomic segments should be provided as strings in the format "chrom:start-end".
        If None, the entire genome will be used.
    step_size
        The step size for sliding the window across the genome (default is None, which uses `window_size`).
    resolution
        The resolution of the stored genomic data (default is 1).
    chroms
        A list of chromosomes to include in the dataset. If None, all chromosomes in the FASTA file will be used.
    temp_dir
        Optional temporary directory for intermediate files. If None, a system temporary directory will be used.

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
    location: PathBuf,
    tmp_dir: TempDir,
    store_builder: Option<DataStoreBuilder>,
}

impl GenomeDataBuilder {
    fn store(&self) -> &DataStoreBuilder {
        self.store_builder
            .as_ref()
            .expect("data store has been moved")
    }

    fn store_mut(&mut self) -> &mut DataStoreBuilder {
        self.store_builder
            .as_mut()
            .expect("data store has been moved")
    }
}

#[pymethods]
impl GenomeDataBuilder {
    #[new]
    #[pyo3(
        signature = (
            location, genome_fasta, window_size, *, segments=None, step_size=None, resolution=1,
            chroms=None, temp_dir=None,
        ),
        text_signature = "($self, location, genome_fasta, window_size, *, segments=None,
            step_size=None, resolution=1, chroms=None, temp_dir=None)"
    )]
    pub fn new(
        location: PathBuf,
        genome_fasta: PathBuf,
        window_size: u32,
        segments: Option<Vec<String>>,
        step_size: Option<u32>,
        resolution: u32,
        chroms: Option<Vec<String>>,
        temp_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let tmp_dir = if let Some(dir) = temp_dir {
            tempfile::tempdir_in(dir)?
        } else {
            tempfile::tempdir()?
        };
        let mut store_builder = DataStoreBuilder::new(
            tmp_dir.as_ref().join("tmp_gdata_builder"),
            window_size,
            resolution,
            0,
        )?;
        let mut fasta_reader = open_fasta(genome_fasta)?;

        // Retrieve chromosome sizes from the FASTA index
        let mut chrom_sizes: BTreeMap<String, u64> = fasta_reader
            .index()
            .as_ref()
            .iter()
            .map(|rec| (rec.name().to_string(), rec.length()))
            .collect();
        if let Some(chroms) = chroms {
            let chroms: HashSet<_> = chroms.into_iter().collect();
            chrom_sizes.retain(|chrom, _| chroms.contains(chrom));
        }

        let segments: Vec<_> = if let Some(s) = segments {
            let mut all_chroms = HashSet::new();
            let s = s
                .into_iter()
                .map(|s| {
                    let mut g = GenomicRange::from_str(&s).unwrap();
                    all_chroms.insert(g.chrom().to_string());
                    expand_segment(&mut g, window_size as u64, &chrom_sizes);
                    g
                })
                .collect();
            chrom_sizes.retain(|chrom, _| all_chroms.contains(chrom));
            s
        } else {
            let step_size = step_size.unwrap_or(window_size);
            get_genome_segments(&chrom_sizes, window_size as u64, step_size as u64)
                .flat_map(|(_, iter)| iter)
                .collect()
        };
        assert!(segments.iter().all_unique(), "segments must be unique");
        store_builder.add_segments(segments, &mut fasta_reader)?;

        Ok(Self {
            location,
            tmp_dir,
            store_builder: Some(store_builder),
        })
    }

    /** Returns the keys (track names) in the dataset.

       This method retrieves all keys from the dataset, which are typically the names of files
       containing genomic data.

       Returns
       -------
       list[str]
           A list of keys as strings.
    */
    pub fn tracks(&self) -> Vec<String> {
        self.store().data_keys.iter().cloned().collect::<Vec<_>>()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    fn segments(&self) -> Vec<String> {
        self.store()
            .segments
            .keys()
            .map(|x| x.pretty_show())
            .collect()
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
    pub fn add_files(&mut self, files: IndexMap<String, PathBuf>) -> Result<()> {
        for (key, path) in files {
            self.add_file(&key, path)?;
        }
        Ok(())
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
        signature = (key, path),
        text_signature = "($self, key, path)",
    )]
    pub fn add_file(&mut self, key: &str, path: PathBuf) -> Result<()> {
        let w5z = W5Z::open(path)?;
        self.store_mut().add_w5z(key, w5z)
    }

    /** Finalizes the dataset creation.

       This method finalizes the dataset by writing all data to the specified location.
       After calling this method, the builder cannot be used to add more data.

       Returns
       -------
       None
    */
    pub fn finish(&mut self) -> Result<()> {
        let store_builder = self
            .store_builder
            .take()
            .expect("data store has been moved");
        store_builder.finish(&self.location)
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
