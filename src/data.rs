use anyhow::{bail, Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use bincode::config::Configuration;
use bincode::{Decode, Encode};
use half::bf16;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayD, Axis};
use noodles::core::Position;
use noodles::fasta::{
    fai::Index,
    io::{indexed_reader::Builder, IndexedReader},
};
use numpy::{PyArray2, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;
use std::{fs::File, io::BufRead, path::Path};

use crate::w5z::W5Z;

/** Represents a builder for genomic data, allowing for the creation and management of genomic datasets.

    This struct provides methods to create a new genomic dataset, open an existing dataset,
    and manage genomic data chunks. It supports operations like adding files, retrieving chromosome
    information, and iterating over data chunks.
*/
#[pyclass]
pub struct GenomeDataBuilder {
    chrom_sizes: BTreeMap<String, u64>,
    window_size: u64,
    resolution: u64,
    location: PathBuf,
}

impl GenomeDataBuilder {
    fn open_(location: impl AsRef<Path>) -> Result<Self> {
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
        let resolution = metadata.get("resolution").unwrap().as_u64().unwrap();
        let window_size = metadata.get("window_size").unwrap().as_u64().unwrap();
        Ok(Self {
            chrom_sizes,
            location: location.as_ref().to_path_buf(),
            window_size,
            resolution,
        })
    }
}

#[pymethods]
impl GenomeDataBuilder {
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, location: PathBuf) -> Result<Self> {
        Self::open_(location)
    }

    #[new]
    #[pyo3(
        signature = (
            location, genome_fasta, *, window_size=524288, step_size=None, chunk_size=128, resolution=1, chroms=None,
        ),
        text_signature = "($self, location, genome_fasta, *, window_size=524288, step_size=None, chunk_size=128, resolution=1, chroms=None)"
    )]
    pub fn new(
        location: PathBuf,
        genome_fasta: PathBuf,
        window_size: u64,
        step_size: Option<u64>,
        chunk_size: usize,
        resolution: u64,
        chroms: Option<Vec<String>>,
    ) -> Result<Self> {
        if window_size % resolution != 0 {
            bail!("Window size must be a multiple of resolution");
        }
        if location.exists() {
            bail!("Location already exists: {}", location.display());
        }
        std::fs::create_dir_all(&location)?;

        let mut reader = open_fasta(genome_fasta)?;
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

        let step_size = step_size.unwrap_or(window_size);
        for (chr, segments) in get_genome_segments(&chrom_sizes, window_size as u64, step_size) {
            for (i, chunk) in segments.chunks(chunk_size).into_iter().enumerate() {
                let (names, seq): (Vec<_>, Vec<_>) = chunk
                    .map(|segment| {
                        let mut s = get_seq(&mut reader, &segment)
                            .unwrap()
                            .sequence()
                            .as_ref()
                            .to_vec();
                        if (segment.len() as u64) < window_size {
                            s.resize(window_size as usize, b'N');
                        }
                        (segment, s)
                    })
                    .unzip();
                let data = DataChunk::new(location.join(&chr).join(i.to_string()), names)?;
                data.save_seqs(seq)?;
            }
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

        Ok(Self {
            chrom_sizes,
            location,
            window_size,
            resolution,
        })
    }

    pub fn chroms(&self) -> Vec<&String> {
        self.chrom_sizes.keys().collect()
    }

    pub fn keys(&self) -> Result<Vec<String>> {
        self.iter_chunks()
            .next()
            .map(|chunk| chunk.keys())
            .unwrap_or_else(|| Ok(vec![]))
    }

    pub fn iter_chunks(&self) -> DataChunkIter {
        let chroms = Box::new(
            self.chrom_sizes
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter(),
        );
        DataChunkIter {
            root: self.location.clone(),
            chroms,
            chunks: Box::new(std::iter::empty()),
        }
    }

    pub fn add_files(&self, files: HashMap<String, PathBuf>) -> Result<()> {
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

    pub fn add_file(&self, key: &str, w5z: PathBuf) -> Result<()> {
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

#[pyclass]
pub struct GenomeDataLoader {
    data: GenomeDataBuilder,
    keys: Vec<String>,
    batch_size: usize,
    buffer_seq: VecDeque<Array1<u8>>,
    buffer_data: VecDeque<ArrayD<f32>>,
    chunks: DataChunkIter,
}

impl Iterator for GenomeDataLoader {
    type Item = (Array2<u8>, ArrayD<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let n_buffer = self.buffer_seq.len();
        if n_buffer < self.batch_size {
            if let Some(chunk) = self.chunks.next() {
                let seqs = chunk.get_seqs().unwrap();
                let values = chunk.gets(&self.keys).unwrap();
                self.buffer_seq.extend(seqs.iter_rows());
                self.buffer_data.extend(values.iter_rows());
                self.next()
            } else {
                if n_buffer == 0 {
                    None
                } else {
                    let r = self.buffer_seq.len();
                    let c = self.buffer_seq[0].len();
                    let seqs = self.buffer_seq.drain(..).flatten().collect::<Vec<_>>();
                    let seqs = Array2::from_shape_vec((r, c), seqs).unwrap();
                    let mut shape: Vec<_> = self.buffer_data[0].shape().to_vec();
                    shape.insert(0, r);
                    let data = self.buffer_data
                            .drain(..)
                            .flatten()
                            .collect::<Vec<_>>();
                    let data = ArrayD::from_shape_vec(shape, data).unwrap();
                    Some((seqs, data))
                }
            }
        } else {
            let c = self.buffer_seq[0].len();
            let seqs = self.buffer_seq.drain(..self.batch_size)
                .flatten()
                .collect::<Vec<_>>();
            let seqs = Array2::from_shape_vec((self.batch_size, c), seqs).unwrap();
            let mut shape = self.buffer_data[0].shape().to_vec();
            shape.insert(0, self.batch_size);
            let data = self.buffer_data
                .drain(..self.batch_size)
                .flatten()
                .collect::<Vec<_>>();
            let data = ArrayD::from_shape_vec(shape, data).unwrap();
            Some((seqs, data))
        }
    }
}

#[pymethods]
impl GenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (
            location, *, batch_size=8,
        ),
        text_signature = "($self, location, *, batch_size=8)"
    )]
    pub fn new(location: PathBuf, batch_size: usize) -> Result<Self> {
        let data = GenomeDataBuilder::open_(location)?;
        let chunks = data.iter_chunks();
        let keys = data.keys()?;
        Ok(Self {
            data,
            batch_size,
            buffer_seq: VecDeque::new(),
            buffer_data: VecDeque::new(),
            chunks,
            keys,
        })
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.chunks = slf.data.iter_chunks();
        slf.buffer_seq.clear();
        slf.buffer_data.clear(); 
        slf
    }

    fn __next__<'a>(mut slf: PyRefMut<'a, Self>, py: Python<'a>) -> Option<(Bound<'a, PyArray2<u8>>, Bound<'a, PyArrayDyn<f32>>)> {
        let (seq, values) = slf.next()?;
        Some((PyArray2::from_owned_array(py, seq), PyArrayDyn::from_owned_array(py, values)))
    }
}

/** Represents an iterator over genomic data chunks, allowing for efficient traversal of genomic datasets.
 
    This struct is used to iterate over genomic data chunks, providing access to each chunk's segments
    and associated data. It supports operations like retrieving the next chunk and iterating over the keys.
*/
#[pyclass]
pub struct DataChunkIter {
    root: PathBuf,
    chroms: Box<dyn Iterator<Item = String> + Send + Sync>,
    chunks: Box<dyn Iterator<Item = PathBuf> + Send + Sync>,
}

impl Iterator for DataChunkIter {
    type Item = DataChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.chunks.next() {
            Some(DataChunk::open(path).unwrap())
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

#[pymethods]
impl DataChunkIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<DataChunk> {
        slf.next()
    }
}

#[derive(Decode, Encode)]
struct Values(#[bincode(with_serde)] ArrayD<bf16>);

impl Values {
    fn iter_rows(&self) -> impl DoubleEndedIterator<Item = ArrayD<f32>> + '_ {
        self.0
            .axis_iter(Axis(0))
            .map(|row| row.mapv(|x| x.to_f32()))
    }
}

#[derive(Decode, Encode)]
struct Sequences(#[bincode(with_serde)] Array2<u8>);

impl Sequences {
    fn iter_rows(&self) -> impl DoubleEndedIterator<Item = Array1<u8>> + '_ {
        self.0.axis_iter(Axis(0)).map(|row| row.to_owned())
    }
}

/** Represents a chunk of genomic data, containing segments and their associated data.
 
    This struct is used to manage genomic data chunks, allowing for efficient storage and retrieval
    of genomic ranges and their associated values. It supports operations like saving sequences,
    retrieving data by keys, and iterating over the keys.
*/
#[pyclass]
pub struct DataChunk {
    location: PathBuf,
    segments: Vec<GenomicRange>,
}

impl DataChunk {
    fn new(location: impl AsRef<Path>, segments: Vec<GenomicRange>) -> Result<Self> {
        std::fs::create_dir_all(location.as_ref().join("data"))?;

        let names = segments.iter().map(|x| x.pretty_show()).join("\n");
        std::fs::write(location.as_ref().join("names.txt"), names)?;
        Ok(Self {
            location: location.as_ref().to_path_buf(),
            segments,
        })
    }

    fn open(location: impl AsRef<Path>) -> Result<Self> {
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
        Ok(Self { location, segments })
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    fn get_seqs(&self) -> Result<Sequences> {
        let seq_file = self.location.join("sequence.dat");
        let seqs = std::fs::read(seq_file)?;
        let mut seqs = zstd::stream::Decoder::new(std::io::Cursor::new(seqs))?;
        let seqs: Sequences = bincode::decode_from_std_read::<_, Configuration, _>(
            &mut seqs,
            Configuration::default(),
        )?;
        Ok(seqs)
    }

    fn get(&self, key: &str) -> Result<Option<Values>> {
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
        Ok(Some(data))
    }

    fn gets(&self, keys: &[String]) -> Result<Values> {
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
        Ok(Values(ndarray::stack(Axis(1), &values)?))
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