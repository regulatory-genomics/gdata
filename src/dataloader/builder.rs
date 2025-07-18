use anyhow::{bail, Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use bincode::config::Configuration;
use bincode::{Decode, Encode};
use half::bf16;
use itertools::Itertools;
use ndarray::{s, Array1, Array2, ArrayD, Axis};
use noodles::core::Position;
use noodles::fasta::{
    fai::Index,
    io::{indexed_reader::Builder, IndexedReader},
};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;
use std::{fs::File, io::BufRead, path::Path};

use crate::dataloader::index::SegmentIndex;
use crate::w5z::W5Z;
use crate::utils::PrefetchIterator;

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
    segments
        Optional list of genomic segments to include in the dataset. If None, the entire genome will be used.
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
*/
#[pyclass]
pub struct GenomeDataBuilder {
    chrom_sizes: BTreeMap<String, u64>,
    window_size: u64,
    resolution: u64,
    location: PathBuf,
    pub(crate) index: SegmentIndex,
}

impl GenomeDataBuilder {
    pub(super) fn open_(location: impl AsRef<Path>) -> Result<Self> {
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
        let index = SegmentIndex::new(location.as_ref(), chrom_sizes.keys());
        let resolution = metadata.get("resolution").unwrap().as_u64().unwrap();
        let window_size = metadata.get("window_size").unwrap().as_u64().unwrap();
        Ok(Self {
            chrom_sizes,
            location: location.as_ref().to_path_buf(),
            window_size,
            resolution,
            index,
        })
    }

    pub(super) fn iter_chunks(&self, buffer_size: usize, trim_target: Option<usize>) -> PrefetchIterator<(Sequences, Values)> {
        let chroms = Box::new(
            self.chrom_sizes
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter(),
        );
        let iter = DataChunkIter {
            root: self.location.clone(),
            chroms,
            chunks: Box::new(std::iter::empty()),
            keys: self.keys().unwrap(),
            trim_target,
        };
        PrefetchIterator::new(iter, buffer_size)
    }
}

#[pymethods]
impl GenomeDataBuilder {
    #[new]
    #[pyo3(
        signature = (
            location, genome_fasta, *, segments=None, window_size=524288, step_size=None, chunk_size=128, resolution=1, chroms=None,
        ),
        text_signature = "($self, location, genome_fasta, *, segments=None, window_size=524288, step_size=None, chunk_size=128, resolution=1, chroms=None)"
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
    ) -> Result<Self> {
        fn write_seqs(
            input: impl IntoIterator<Item = (String, impl IntoIterator<Item = GenomicRange>)>,
            location: &PathBuf,
            fasta_reader: &mut IndexedReader<noodles::fasta::io::BufReader<File>>,
            window_size: u64,
            chunk_size: usize,
        ) -> Result<()>
        {
            for (chr, segments) in input {
                for (i, chunk) in segments.into_iter().chunks(chunk_size).into_iter().enumerate() {
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
            bail!("Location already exists: {}", location.display());
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
                    let g = GenomicRange::from_str(&s).map_err(|_| { anyhow::anyhow!("Failed to parse segment '{}'", s)})?;
                    all_chroms.insert(g.chrom().to_string());
                    Ok(g)
                })
                .collect::<Result<Vec<_>>>()?;
            segments.sort();
            write_seqs(&segments.into_iter().chunk_by(|x| x.chrom().to_string()), &location, &mut reader, window_size, chunk_size)?;
            chrom_sizes.retain(|chrom, _| all_chroms.contains(chrom));
        } else {
            let step_size = step_size.unwrap_or(window_size);
            write_seqs(get_genome_segments(&chrom_sizes, window_size as u64, step_size), &location, &mut reader, window_size, chunk_size)?;
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

        let index = SegmentIndex::new(&location, chrom_sizes.keys());
        Ok(Self {
            chrom_sizes,
            location,
            window_size,
            resolution,
            index,
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
        signature = (location),
        text_signature = "(location)",
    )]
    fn open(_cls: &Bound<'_, PyType>, location: PathBuf) -> Result<Self> {
        Self::open_(location)
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
                let mut result  = std::fs::read_dir(entry?.path().join("data"))?
                    .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
                    .collect::<Result<Vec<_>>>()?;
                result.sort();
                Ok(result)
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        }
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
    fn add_files(&self, files: HashMap<String, PathBuf>) -> Result<()> {
        files
            .into_iter()
            .chunks(64)
            .into_iter()
            .try_for_each(|chunk| {
                chunk
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .try_for_each(|(key, w5z)| self.add_file(&key, w5z))
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
    fn add_file(&self, key: &str, w5z: PathBuf) -> Result<()> {
        let n_cols = self.window_size / self.resolution;
        let w5z = W5Z::open(w5z)?;
        for chr in self.chroms() {
            let chunks: Vec<_> = std::fs::read_dir(self.location.join(&chr))?
                .filter_map(Result::ok)
                .collect();
            let values = w5z.get(&chr)?.to_vec();
            for c in chunks {
                let data = DataChunk::open(c.path())?;
                let arr_elems: Vec<_> = data
                    .segments
                    .iter()
                    .flat_map(|segment| {
                        let start = segment.start() as usize;
                        let end = segment.end() as usize;
                        let mut v: Vec<_> = if self.resolution > 1 {
                            values[start..end]
                                .chunks(self.resolution as usize)
                                .map(|x| {
                                    let m: average::Mean = x.iter().map(|x| *x as f64).collect();
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
                let arr =
                    Array2::from_shape_vec((data.segments.len(), n_cols as usize), arr_elems)?
                        .into_dyn();
                data.save_data(key, &Values(arr))?;
            }
        }
        Ok(())
    }
}

/** Represents an iterator over genomic data chunks, allowing for efficient traversal of genomic datasets.
 
    This struct is used to iterate over genomic data chunks, providing access to each chunk's segments
    and associated data. It supports operations like retrieving the next chunk and iterating over the keys.
*/
struct DataChunkIter {
    root: PathBuf,
    chroms: Box<dyn Iterator<Item = String> + Send + Sync>,
    chunks: Box<dyn Iterator<Item = PathBuf> + Send + Sync>,
    keys: Vec<String>,
    trim_target: Option<usize>,
}

impl Iterator for DataChunkIter {
    type Item = (Sequences, Values);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.chunks.next() {
            let mut chunk = DataChunk::open(path).unwrap();
            if let Some(trim_target) = self.trim_target {
                chunk.set_trim_target(trim_target);
            }
            let seqs = chunk.get_seqs().unwrap();
            let values = chunk.gets(&self.keys).unwrap();
            Some((seqs, values))
        } else {
            let chr = self.chroms.next()?;
            self.chunks = Box::new(
                std::fs::read_dir(self.root.join(&chr))
                    .unwrap()
                    .map(|entry| entry.unwrap().path()),
            );
            self.next()
        }
    }
}

#[derive(Decode, Encode)]
pub struct Values(#[bincode(with_serde)] pub ArrayD<bf16>);

impl Values {
    pub(super) fn iter_rows(&self) -> impl DoubleEndedIterator<Item = ArrayD<f32>> + '_ {
        self.0
            .axis_iter(Axis(0))
            .map(|row| row.mapv(|x| x.to_f32()))
    }
}

#[derive(Decode, Encode)]
pub struct Sequences(#[bincode(with_serde)] pub Array2<u8>);

impl Sequences {
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = Array1<u8>> + '_ {
        self.0.axis_iter(Axis(0)).map(|row| row.to_owned())
    }
}

/** Represents a chunk of genomic data, containing segments and their associated data.
 
    This struct is used to manage genomic data chunks, allowing for efficient storage and retrieval
    of genomic ranges and their associated values. It supports operations like saving sequences,
    retrieving data by keys, and iterating over the keys.
*/
#[pyclass]
#[derive(Clone, Debug)]
pub struct DataChunk {
    location: PathBuf,
    pub(crate) segments: Vec<GenomicRange>,
    trim_target: Option<usize>,
}

impl DataChunk {
    fn new(location: impl AsRef<Path>, segments: Vec<GenomicRange>) -> Result<Self> {
        std::fs::create_dir_all(location.as_ref().join("data"))?;

        let names = segments.iter().map(|x| x.pretty_show()).join("\n");
        std::fs::write(location.as_ref().join("names.txt"), names)?;
        Ok(Self {
            location: location.as_ref().to_path_buf(),
            segments,
            trim_target: None,
        })
    }

    pub fn open(location: impl AsRef<Path>) -> Result<Self> {
        let location = location.as_ref().to_path_buf();
        if !location.exists() {
            return Err(anyhow::anyhow!(
                "Data chunk not found at {}",
                location.display()
            ));
        }
        let segments: Vec<GenomicRange> = std::fs::read_to_string(location.join("names.txt"))?
            .lines()
            .map(|line| GenomicRange::from_str(line).unwrap())
            .collect();
        Ok(Self { location, segments, trim_target: None })
    }

    fn set_trim_target(&mut self, trim_target: usize) {
        self.trim_target = Some(trim_target);
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn get_seqs(&self) -> Result<Sequences> {
        let seq_file = self.location.join("sequence.dat");
        let seqs = std::fs::read(seq_file)?;
        let mut seqs = zstd::stream::Decoder::new(std::io::Cursor::new(seqs))?;
        let seqs: Sequences = bincode::decode_from_std_read::<_, Configuration, _>(
            &mut seqs,
            Configuration::default(),
        )?;
        Ok(seqs)
    }

    pub fn get_seq_at(&self, i: usize) -> Result<Array1<u8>> {
        if i >= self.segments.len() {
            bail!("Index out of bounds: {} >= {}", i, self.segments.len());
        }
        let seqs = self.get_seqs()?;
        let seq = seqs.iter_rows().nth(i).ok_or_else(|| {
            anyhow::anyhow!("Failed to get sequence at index {}", i)
        })?;
        Ok(seq)
    }

    pub fn get(&self, key: &str) -> Result<Option<Values>> {
        let path = self.location.join("data").join(key);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read(path)?;
        let mut data = zstd::stream::Decoder::new(std::io::Cursor::new(data))?;
        let data: Values = bincode::decode_from_std_read::<_, Configuration, _>(
            &mut data,
            Configuration::default(),
        )?;
        if let Some(trim_target) = self.trim_target {
            let size = data.0.shape()[1];
            let data = data.0.slice(s![.., trim_target..size - trim_target]);
            Ok(Some(Values(data.into_dyn().to_owned())))
        } else {
            Ok(Some(data))
        }
    }

    pub fn gets(&self, keys: &[String]) -> Result<Values> {
        let values = keys.par_iter()
            .map(|key| {
                let data = self.get(key)?.with_context(|| {
                    format!(
                        "Failed to get key '{}' from chunk at {}",
                        key,
                        self.location.display()
                    )
                })?;
                Ok(data.0)
            })
            .collect::<Result<Vec<_>>>()?;
        let values = values
            .iter()
            .map(|arr| arr.view())
            .collect::<Vec<_>>();
        Ok(Values(ndarray::stack(Axis(2), &values)?))
    }

    fn save_seqs(&self, seqs: Vec<Vec<u8>>) -> Result<()> {
        let shape = (seqs.len(), seqs[0].len());
        let seqs: Result<Vec<u8>> = seqs.into_iter().flatten()
            .map(|x| encode_nucleotide(x)).collect();
        let seqs = Sequences(Array2::from_shape_vec(shape, seqs?)?);
        let seqs = bincode::encode_to_vec::<_, Configuration>(seqs, Default::default())?;
        let seqs = zstd::bulk::Compressor::new(7)?.compress(&seqs)?;
        std::fs::write(self.location.join("sequence.dat"), seqs)?;
        Ok(())
    }

    fn save_data(&self, key: &str, data: &Values) -> Result<()> {
        let data = bincode::encode_to_vec::<_, Configuration>(data, Configuration::default())?;
        let data = zstd::bulk::Compressor::new(7)?.compress(&data)?;
        std::fs::write(self.location.join("data").join(key), data)?;
        Ok(())
    }
}

/*
#[pymethods]
impl DataChunk {
    fn keys(&self) -> Result<Vec<String>> {
        std::fs::read_dir(self.location.join("data"))?
            .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
            .collect()
    }

    /// Returns the dna sequences stored in this data chunk.
    /// The nucleotides are encoded as integers:
    /// - A: 0
    /// - C: 1
    /// - G: 2
    /// - T: 3
    /// - N: 4
    fn sequences<'a>(&'a self, py: Python<'a>) -> Result<Bound<'a, PyArray2<u8>>> {
        let seqs = self.get_seqs()?;
        Ok(PyArray2::from_owned_array(py, seqs.0))
    }

    /// Retrieves the values associated with a key or keys from the data chunk.
    fn __getitem__<'a>(&'a self, py: Python<'a>, key: Bound<'_, PyAny>) -> Result<Bound<'a, PyArrayDyn<f32>>> {
        let arr = if let Ok(key) = key.extract::<String>() {
            self
                .get(&key)?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Key '{}' not found in chunk at {}",
                        key,
                        self.location.display()
                    )
                })
                .map(|values| values.0)?
        } else {
            let key: Vec<String> = key.extract()?;
            self.gets(key.as_slice())?.0
        };
        Ok(PyArrayDyn::from_owned_array(py, arr.mapv(|x| x.to_f32())))
    }

    fn __len__(&self) -> usize {
        self.segments.len()
    }

    fn __repr__(&self) -> String {
        format!("DataChunk at '{}'", self.location.display())
    }
}
*/

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

fn open_fasta(genome_fasta: impl AsRef<Path>) -> Result<IndexedReader<noodles::fasta::io::BufReader<File>>> {
    let fai = PathBuf::from(format!("{}.fai", genome_fasta.as_ref().display()));
    let index = if fai.exists() {
        noodles::fasta::fai::io::Reader::new(BufReader::new(File::open(&fai)?))
            .read_index()
            .context("Failed to read FASTA index")?
    } else {
        let reader =
            noodles::fasta::io::reader::Builder::default().build_from_path(&genome_fasta)?;
        create_index(reader.into_inner())?
    };

    let reader = Builder::default().set_index(index).build_from_path(&genome_fasta)?;
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

fn encode_nucleotide(base: u8) -> Result<u8> {
    let b = match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        b'N' | b'n' => 4,
        _ => bail!("Invalid DNA base: {}", base as char),
    };
    Ok(b)
}