use anyhow::{bail, Context, Result};
use bed_utils::bed::GenomicRange;
use bincode::config::Configuration;
use bincode::{Decode, Encode};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array1, Array2, ArrayD, Axis};
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::io::{Read, Seek, Write};
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

/** Represents an iterator over genomic data chunks, allowing for efficient traversal of genomic datasets.

    This struct is used to iterate over genomic data chunks, providing access to each chunk's segments
    and associated data. It supports operations like retrieving the next chunk and iterating over the keys.
*/
pub(crate) struct DataChunkIter {
    pub(crate) root: PathBuf,
    pub(crate) chroms: Box<dyn Iterator<Item = String> + Send + Sync>,
    pub(crate) chunks: Box<dyn Iterator<Item = PathBuf> + Send + Sync>,
    pub(crate) trim_target: Option<usize>,
}

impl Iterator for DataChunkIter {
    type Item = DataChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.chunks.next() {
            let mut chunk = DataChunk::open(path).unwrap();
            if let Some(trim_target) = self.trim_target {
                chunk.set_trim_target(trim_target);
            }
            Some(chunk)
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

    fn trim(self, trim_target: usize) -> Self {
        let size = self.0.shape()[1];
        let data = self.0.slice(s![.., trim_target..size - trim_target]);
        Values(data.into_dyn().to_owned())
    }

    fn decode(buffer: &[u8]) -> Result<Self> {
        let mut decoder = zstd::stream::Decoder::new(std::io::Cursor::new(buffer))?;
        let data: Values = bincode::decode_from_std_read::<_, Configuration, _>(
            &mut decoder,
            Configuration::default(),
        )?;
        Ok(data)
    }

    fn decode_many(buffers: Vec<Vec<u8>>, trim_target: Option<usize>) -> Result<Self> {
        let values = buffers
            .into_par_iter()
            .map(|buffer| {
                let data = Values::decode(&buffer)?;
                if let Some(trim_target) = trim_target {
                    Ok(data.trim(trim_target))
                } else {
                    Ok(data)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let values = values.iter().map(|arr| arr.0.view()).collect::<Vec<_>>();
        Ok(Values(ndarray::stack(Axis(2), &values)?))
    }
}

#[derive(Decode, Encode)]
pub struct Sequences(#[bincode(with_serde)] pub Array2<u8>);

impl Sequences {
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = Array1<u8>> + '_ {
        self.0.axis_iter(Axis(0)).map(|row| row.to_owned())
    }

    fn decode(buffer: &[u8]) -> Result<Self> {
        let mut seqs = zstd::stream::Decoder::new(std::io::Cursor::new(buffer))?;
        let seqs: Sequences = bincode::decode_from_std_read::<_, Configuration, _>(
            &mut seqs,
            Configuration::default(),
        )?;
        Ok(seqs)
    }
}

/** Represents a chunk of genomic data, containing segments and their associated data.

    This struct is used to manage genomic data chunks, allowing for efficient storage and retrieval
    of genomic ranges and their associated values. It supports operations like saving sequences,
    retrieving data by keys, and iterating over the keys.
*/
#[pyclass]
#[derive(Debug)]
pub struct DataChunk {
    location: PathBuf,
    pub(crate) segments: Vec<GenomicRange>,
    trim_target: Option<usize>,
    data_store: Option<DataStore>,
}

impl DataChunk {
    pub fn new(location: impl AsRef<Path>, segments: Vec<GenomicRange>) -> Result<Self> {
        std::fs::create_dir_all(location.as_ref().join("data"))?;

        let names = segments.iter().map(|x| x.pretty_show()).join("\n");
        std::fs::write(location.as_ref().join("names.txt"), names)?;
        Ok(Self {
            location: location.as_ref().to_path_buf(),
            segments,
            trim_target: None,
            data_store: None,
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
        let store_path = location.join("_data");
        let data_store = if store_path.exists() {
            Some(DataStore::open(store_path)?)
        } else {
            None
        };
        Ok(Self {
            location,
            segments,
            trim_target: None,
            data_store,
        })
    }

    pub fn set_trim_target(&mut self, trim_target: usize) {
        self.trim_target = Some(trim_target);
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn keys(&self) -> Result<Vec<String>> {
        if let Some(data_store) = &self.data_store {
            Ok(data_store.index.keys().cloned().collect())
        } else {
            let data_dir = self.location.join("data");
            let mut result = std::fs::read_dir(data_dir)?
                .map(|entry| {
                    let entry = entry?;
                    Ok(entry.file_name().into_string().unwrap())
                })
                .collect::<Result<Vec<_>>>()?;
            result.sort();
            Ok(result)
        }
    }

    pub fn get_seqs(&self) -> Result<Sequences> {
        let seq_file = self.location.join("sequence.dat");
        let seqs = std::fs::read(seq_file)?;
        Sequences::decode(&seqs)
    }

    pub fn get_seq_at(&self, i: usize) -> Result<Array1<u8>> {
        if i >= self.segments.len() {
            bail!("Index out of bounds: {} >= {}", i, self.segments.len());
        }
        let seqs = self.get_seqs()?;
        let seq = seqs
            .iter_rows()
            .nth(i)
            .ok_or_else(|| anyhow::anyhow!("Failed to get sequence at index {}", i))?;
        Ok(seq)
    }

    fn data_store(&self) -> Result<&DataStore> {
        self.data_store.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Data store not initialized. Please call `finish()` to create the data store."
            )
        })
    }

    pub fn get(&self, key: &str) -> Result<Option<Values>> {
        let data = self.data_store()?.read(key)?;
        if let Some(trim_target) = self.trim_target {
            Ok(data.map(|x| x.trim(trim_target)))
        } else {
            Ok(data)
        }
    }

    pub fn gets(&self, keys: &[String]) -> Result<Values> {
        self.data_store()?.read_many(keys, self.trim_target)
    }

    pub fn read_all(&self) -> Result<Values> {
        self.data_store()?.read_all(self.trim_target)
    }

    pub fn save_seqs(&self, seqs: Vec<Vec<u8>>) -> Result<()> {
        let shape = (seqs.len(), seqs[0].len());
        let seqs: Result<Vec<u8>> = seqs
            .into_iter()
            .flatten()
            .map(|x| encode_nucleotide(x))
            .collect();
        let seqs = Sequences(Array2::from_shape_vec(shape, seqs?)?);
        let seqs = bincode::encode_to_vec::<_, Configuration>(seqs, Default::default())?;
        let seqs = zstd::bulk::Compressor::new(9)?.compress(&seqs)?;
        std::fs::write(self.location.join("sequence.dat"), seqs)?;
        Ok(())
    }

    pub fn save_data(&self, key: &str, data: &Values) -> Result<()> {
        let data = bincode::encode_to_vec::<_, Configuration>(data, Configuration::default())?;
        let data = zstd::bulk::Compressor::new(9)?.compress(&data)?;
        std::fs::write(self.location.join("data").join(key), data)?;
        Ok(())
    }

    /// Consolidates the data chunk by merging data files into a single file.
    pub fn consolidate(&mut self) -> Result<()> {
        let mut data_files = std::fs::read_dir(self.location.join("data"))?
            .map(|entry| {
                let entry = entry?;
                Ok((entry.file_name().into_string().unwrap(), entry.path()))
            })
            .collect::<Result<Vec<_>>>()?;
        data_files.sort_by(|a, b| a.0.cmp(&b.0));
        let data_store = DataStore::create(self.location.join("_data"), data_files)?;
        self.data_store = Some(data_store);
        std::fs::remove_dir_all(self.location.join("data"))?;
        Ok(())
    }
}

#[derive(Debug)]
struct DataStore {
    index: IndexMap<String, (usize, usize)>,
    file: PathBuf,
}

impl DataStore {
    fn create(
        location: impl AsRef<Path>,
        data: impl IntoIterator<Item = (String, impl AsRef<Path>)>,
    ) -> Result<Self> {
        let mut file = std::fs::File::create(&location)?;
        let mut index = IndexMap::new();
        let mut buffer = Vec::new();
        let mut acc = 0;

        for (key, fl) in data {
            buffer.clear();
            let n = std::fs::File::open(fl)?.read_to_end(&mut buffer)?;
            file.write_all(buffer.as_slice())?;
            index.insert(key.to_string(), (acc, n));
            acc += n;
        }

        file.flush()?;
        let index_file = location.as_ref().with_extension("index");
        let index_data = bincode::serde::encode_to_vec(&index, bincode::config::standard())?;
        std::fs::write(index_file, index_data)?;
        file.sync_all()?;
        Ok(Self {
            index,
            file: location.as_ref().to_path_buf(),
        })
    }

    fn open(location: impl AsRef<Path>) -> Result<Self> {
        let index_file = location.as_ref().with_extension("index");
        if !index_file.exists() {
            bail!("Index file not found at {}", index_file.display());
        }
        let index_data = std::fs::read(index_file)?;
        let index: IndexMap<String, (usize, usize)> =
            bincode::serde::decode_from_slice(&index_data, bincode::config::standard())?.0;
        Ok(Self {
            index,
            file: location.as_ref().to_path_buf(),
        })
    }

    fn read(&self, key: &str) -> Result<Option<Values>> {
        if !self.index.contains_key(key) {
            return Ok(None);
        }

        let (offset, size) = self.index.get(key).unwrap();
        let mut file = std::fs::File::open(&self.file).with_context(|| {
            format!("Failed to open data store file at {}", self.file.display())
        })?;
        file.seek(std::io::SeekFrom::Start(*offset as u64))?;
        let mut buffer = vec![0; *size];
        file.read_exact(&mut buffer)?;
        Ok(Some(Values::decode(&buffer)?))
    }

    fn read_many(&self, keys: &[String], trim_target: Option<usize>) -> Result<Values> {
        let mut file = std::fs::File::open(&self.file).with_context(|| {
            format!("Failed to open data store file at {}", self.file.display())
        })?;
        let bytes = keys
            .iter()
            .map(|key| {
                let (offset, size) = self.index.get(key).unwrap();
                let mut buffer = vec![0; *size];
                file.seek(std::io::SeekFrom::Start(*offset as u64))?;
                file.read_exact(&mut buffer)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>>>()?;

        Values::decode_many(bytes, trim_target)
    }

    fn read_all(&self, trim_target: Option<usize>) -> Result<Values> {
        let mut file = std::fs::File::open(&self.file).with_context(|| {
            format!("Failed to open data store file at {}", self.file.display())
        })?;

        let raw_bytes = self
            .index
            .iter()
            .map(|(_, (_, size))| {
                let mut buffer = vec![0; *size];
                file.read_exact(&mut buffer)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>>>()?;

        Values::decode_many(raw_bytes, trim_target)
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

pub(crate) fn encode_nucleotide(base: u8) -> Result<u8> {
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

pub(crate) fn decode_nucleotide(base: u8) -> Result<u8> {
    let b = match base {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        4 => b'N',
        _ => bail!("Invalid DNA base: {}", base),
    };
    Ok(b)
}
