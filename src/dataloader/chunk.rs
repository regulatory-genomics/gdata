use anyhow::{bail, Context, Result};
use bed_utils::bed::GenomicRange;
use bincode::config::Configuration;
use bincode::{Decode, Encode};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array1, Array2, ArrayD, Axis};
use numpy::PyArray2;
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
    pub(crate) mutable: bool,
}

impl Iterator for DataChunkIter {
    type Item = DataChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.chunks.next() {
            let mut chunk = DataChunk::open(path, self.mutable).unwrap();
            if let Some(trim_target) = self.trim_target {
                chunk.set_trim_target(trim_target);
            }
            Some(chunk)
        } else {
            let chr = self.chroms.next()?;
            self.chunks = Box::new(
                std::fs::read_dir(self.root.join(&chr))
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .sorted_by_key(|p| p.file_name().unwrap().to_string_lossy().parse::<usize>().unwrap())
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

    /// Perform in-place scaling and clamping on the values.
    fn transform(&mut self, scale: Option<bf16>, clamp_max: Option<bf16>) {
        self.0.map_inplace(|x| {
            if let Some(scale) = scale {
                *x *= scale;
            }
            if let Some(clamp_max) = clamp_max {
                if *x > clamp_max {
                    *x = clamp_max;
                }
            }
        })
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

    fn encode(self) -> Result<Vec<u8>> {
        let data = bincode::encode_to_vec::<_, Configuration>(self, Configuration::default())?;
        let data = zstd::bulk::Compressor::new(9)?.compress(&data)?;
        Ok(data)
    }
}

#[derive(Debug, Decode, Encode, PartialEq)]
pub struct Sequences(#[bincode(with_serde)] pub Array2<u8>);

impl<'py> IntoPyObject<'py> for Sequences {
    type Target = PyArray2<u8>;
    type Output = Bound<'py, Self::Target>;
    type Error = anyhow::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyArray2::from_owned_array(py, self.0))
    }
}

impl Sequences {
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = Array1<u8>> + '_ {
        self.0.axis_iter(Axis(0)).map(|row| row.to_owned())
    }

    pub fn into_strings(self) -> Vec<String> {
        self.0.rows()
            .into_iter()
            .map(|row| {
                let row: Vec<_> = row.iter().map(|x| decode_nucleotide(*x).unwrap()).collect();
                String::from_utf8(row).unwrap()
            })
            .collect::<Vec<_>>()
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
    data_store: DataStore,
}

impl DataChunk {
    pub fn new(location: impl AsRef<Path>, segments: Vec<GenomicRange>) -> Result<Self> {
        std::fs::create_dir_all(location.as_ref())?;
        let names = segments.iter().map(|x| x.pretty_show()).join("\n");
        std::fs::write(location.as_ref().join("names.txt"), names)?;
        Ok(Self {
            location: location.as_ref().to_path_buf(),
            segments,
            trim_target: None,
            data_store: DataStore::create(
                location.as_ref().join("data"),
                std::iter::empty::<(_, String)>(),
            )?,
        })
    }

    pub fn open(location: impl AsRef<Path>, writable: bool) -> Result<Self> {
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
        let store_path = location.join("data");
        Ok(Self {
            location,
            segments,
            trim_target: None,
            data_store: DataStore::open(store_path, writable)?,
        })
    }

    pub fn set_trim_target(&mut self, trim_target: usize) {
        self.trim_target = Some(trim_target);
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data_store.index.keys()
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

    pub fn get(&mut self, key: &str) -> Result<Option<Values>> {
        let data = self.data_store.read(key)?;
        if let Some(trim_target) = self.trim_target {
            Ok(data.map(|x| x.trim(trim_target)))
        } else {
            Ok(data)
        }
    }

    pub fn gets(&mut self, keys: &[String]) -> Result<Values> {
        self.data_store.read_many(keys, self.trim_target)
    }

    pub fn read_all(&mut self) -> Result<Values> {
        self.data_store.read_all(self.trim_target)
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

    pub fn save_data(
        &mut self,
        data: impl IntoParallelIterator<Item = (String, Values)>,
    ) -> Result<()> {
        self.data_store.write_par(data)
    }
}

#[derive(Debug)]
struct DataStore {
    index: IndexMap<String, (usize, usize)>,
    file: std::fs::File,
    location: PathBuf,
}

impl DataStore {
    fn create(
        location: impl AsRef<Path>,
        data: impl IntoIterator<Item = (String, impl AsRef<Path>)>,
    ) -> Result<Self> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .append(true)
            .create(true)
            .open(&location)
            .with_context(|| {
                format!(
                    "Failed to create data file at {}",
                    location.as_ref().display()
                )
            })?;

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
        let store = Self {
            index,
            file,
            location: location.as_ref().to_path_buf(),
        };
        store.write_index()?;
        Ok(store)
    }

    fn write_index(&self) -> Result<()> {
        let index_file = self.location.with_extension("index");
        let index_data = bincode::serde::encode_to_vec(&self.index, bincode::config::standard())?;
        std::fs::write(index_file, index_data)?;
        Ok(())
    }

    fn open(location: impl AsRef<Path>, writable: bool) -> Result<Self> {
        let mut opt = std::fs::OpenOptions::new();
        let opt = opt.create(false).read(true);
        let opt = if writable {
            opt.write(true).append(true)
        } else {
            opt.write(false)
        };
        let file = opt.open(&location).with_context(|| {
            format!(
                "Failed to open data file at {}",
                location.as_ref().display()
            )
        })?;

        let index_file = location.as_ref().with_extension("index");
        if !index_file.exists() {
            bail!("Index file not found at {}", index_file.display());
        }
        let index_data = std::fs::read(index_file)?;
        let index: IndexMap<String, (usize, usize)> =
            bincode::serde::decode_from_slice(&index_data, bincode::config::standard())?.0;
        Ok(Self {
            index,
            file,
            location: location.as_ref().to_path_buf(),
        })
    }

    fn read(&mut self, key: &str) -> Result<Option<Values>> {
        if !self.index.contains_key(key) {
            return Ok(None);
        }

        let (offset, size) = self.index.get(key).unwrap();
        self.file.seek(std::io::SeekFrom::Start(*offset as u64))?;
        let mut buffer = vec![0; *size];
        self.file.read_exact(&mut buffer)?;
        Ok(Some(Values::decode(&buffer)?))
    }

    fn read_many(&mut self, keys: &[String], trim_target: Option<usize>) -> Result<Values> {
        let bytes = keys
            .iter()
            .map(|key| {
                let (offset, size) = self.index.get(key).unwrap();
                let mut buffer = vec![0; *size];
                self.file.seek(std::io::SeekFrom::Start(*offset as u64))?;
                self.file.read_exact(&mut buffer)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>>>()?;

        Values::decode_many(bytes, trim_target)
    }

    fn read_all(&mut self, trim_target: Option<usize>) -> Result<Values> {
        self.file.rewind()?;
        let raw_bytes = self
            .index
            .iter()
            .map(|(_, (_, size))| {
                let mut buffer = vec![0; *size];
                self.file.read_exact(&mut buffer)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>>>()?;

        Values::decode_many(raw_bytes, trim_target)
    }

    fn write_par(
        &mut self,
        data: impl IntoParallelIterator<Item = (String, Values)>,
    ) -> Result<()> {
        let mut offset = self.file.seek(std::io::SeekFrom::End(0))? as usize;

        let data: Vec<_> = data
            .into_par_iter()
            .map(|(key, values)| (key, values.encode().unwrap()))
            .collect();

        data.into_iter().for_each(|(key, buffer)| {
            let size = buffer.len();
            self.file.write_all(&buffer).unwrap();
            self.index.insert(key, (offset, size));
            offset += size;
        });

        self.file.flush()?;
        self.write_index()?;
        Ok(())
    }
}

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
