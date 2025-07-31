use anyhow::{bail, Context, Result};
use bed_utils::bed::GenomicRange;
use bincode::config::Configuration;
use bincode::{Decode, Encode};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array1, Array2, Array3, ArrayD, Axis};
use numpy::{Ix2, PyArray2};
use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Decode, Encode)]
pub struct Values(#[bincode(with_serde)] pub Array3<bf16>);

impl Values {
    /// Split the values into consecutive chunks on the second dimension (the sequence).
    /// The last chunk is dropped if it is smaller.
    pub fn split(self, size: usize) -> Result<Self> {
        let (d, h, w) = self.0.dim();

        if size == 0 || h % size != 0 {
            bail!(
                "Cannot split values into chunks of size {}: length {} is not a multiple of size",
                size,
                h
            );
        }

        let num_chunks = h / size;

        // Reshape to 4D to expose the chunks as a new dimension.
        // The shape becomes (d, num_chunks, chunk_height, w).
        // No axis permutation is needed because the dimensions are already in the correct order
        // to be collapsed.
        let intermediate_shape = (d, num_chunks, size, w);

        // Reshape again to the final 3D shape by collapsing the first two dimensions.
        let final_shape = (d * num_chunks, size, w);

        let result = self
            .0
            .into_shape_with_order(intermediate_shape)?
            .into_shape_with_order(final_shape)?;

        Ok(Values(result))
    }

    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = Array2<f32>> + '_ {
        self.0
            .axis_iter(Axis(0))
            .map(|row| row.mapv(|x| x.to_f32()))
    }

    pub fn select_rows(&self, indices: &[usize]) -> Self {
        Values(self.0.select(Axis(0), indices))
    }

    /// Perform in-place scaling and clamping on the values.
    pub fn transform(&mut self, scale: Option<bf16>, clamp_max: Option<bf16>) {
        self.0.map_inplace(|x| {
            if x.is_nan() {
                *x = bf16::from_f32(0.0); // Replace NaN with 0.0
            } else {
                if let Some(scale) = scale {
                    *x *= scale;
                }
                if let Some(clamp_max) = clamp_max {
                    if *x > clamp_max {
                        *x = clamp_max;
                    }
                }
            }
        })
    }

    fn decode(buffer: &[u8]) -> Result<Self> {
        let data = decompress_data_zst(buffer);
        let data: Values =
            bincode::decode_from_slice::<_, Configuration>(&data, Configuration::default())?.0;
        let data = data.0.permuted_axes([1, 2, 0]); // Permute the axes back
        Ok(Values(data.as_standard_layout().to_owned()))
    }

    fn encode(self) -> Result<Vec<u8>> {
        // Permute the axes to offer better compression ratio, as the second axis
        // is least variable, followed by the first, and the third is most variable.
        let data = bincode::encode_to_vec::<_, Configuration>(
            Values(self.0.permuted_axes([2, 0, 1])),
            Configuration::default(),
        )?;
        let data = compress_data_zst(data);
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
    pub fn split(self, size: usize) -> Result<Self> {
        let (d, h) = self.0.dim();

        if size == 0 || h % size != 0 {
            bail!(
                "Cannot split values into chunks of size {}: length {} is not a multiple of size",
                size,
                h
            );
        }

        let num_chunks = h / size;
        let intermediate_shape = (d, num_chunks, size);
        let final_shape = (d * num_chunks, size);

        let result = self
            .0
            .into_shape_with_order(intermediate_shape)?
            .into_shape_with_order(final_shape)?;

        Ok(Self(result))
    }

    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = Array1<u8>> + '_ {
        self.0.axis_iter(Axis(0)).map(|row| row.to_owned())
    }

    pub fn select_rows(&self, indices: &[usize]) -> Self {
        Self(self.0.select(Axis(0), indices))
    }

    pub fn into_strings(self) -> Vec<String> {
        self.0
            .rows()
            .into_iter()
            .map(|row| {
                let row: Vec<_> = row.iter().map(|x| decode_nucleotide(*x).unwrap()).collect();
                String::from_utf8(row).unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn decode(buffer: &[u8]) -> Result<Self> {
        let mut seqs = decompress_data_zst(buffer);
        let seqs: Sequences =
            bincode::decode_from_slice::<_, Configuration>(&mut seqs, Configuration::default())?.0;
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
    location: PathBuf,                      // Path to the chunk's directory
    pub(crate) segments: Vec<GenomicRange>, // Genomic segments contained in this chunk
    trim_target: Option<usize>,             // Output trimming
    split_data: Option<(usize, usize)>,     // Optional size for splitting data
    data_store: DataStore,                  // Data store
    subset: Option<Vec<usize>>,             // Optional indices representing a subset of the data
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
            split_data: None,
            data_store: DataStore::create(
                location.as_ref().join("data"),
                std::iter::empty::<(_, String)>(),
            )?,
            subset: None,
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
            split_data: None,
            data_store: DataStore::open(store_path, writable)?,
            subset: None,
        })
    }

    pub fn set_trim_target(&mut self, trim_target: usize) {
        self.trim_target = Some(trim_target);
    }

    pub fn set_split_data(&mut self, split_data: (usize, usize)) {
        self.split_data = Some(split_data);
    }

    /// indices must be unique and sorted
    pub(crate) fn subset(&mut self, indices: Vec<usize>) -> Result<()> {
        if self.subset.is_some() {
            bail!("Subset has already been set for this DataChunk");
        }
        if indices.is_empty() {
            bail!("Cannot create subset with empty indices");
        }

        let segments: Vec<_> = indices.iter().map(|&i| self.segments[i].clone()).collect();
        if segments.len() != self.segments.len() {
            self.segments = segments;
            self.subset = Some(indices);
        }

        Ok(())
    }

    /// Returns the chunk size, i.e., the number of records in the chunk.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data_store.index.keys()
    }

    pub fn get_seqs(&self) -> Result<Sequences> {
        let seq_file = self.location.join("sequence.dat");
        let mut seqs = Sequences::decode(&std::fs::read(seq_file)?)?;
        if let Some(idx) = self.subset.as_ref() {
            seqs = seqs.select_rows(idx);
        }
        if let Some(split_data) = self.split_data {
            seqs = seqs.split(split_data.0)?;
        }
        Ok(seqs)
    }

    pub(crate) fn get_seq_at(&self, i: usize) -> Result<Array1<u8>> {
        if self.split_data.is_some() {
            bail!("Cannot get sequence at index {} when split_data is set", i);
        }

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

    pub fn read_keys(&mut self, keys: &[String]) -> Result<Values> {
        let mut data = self.data_store.read_keys(keys);
        if let Some(idx) = self.subset.as_ref() {
            data = data.select_rows(idx);
        }
        if let Some(split_data) = self.split_data {
            data = data.split(split_data.1)?;
        }
        let mut arr = data.0;
        if let Some(trim_target) = self.trim_target {
            arr = arr
                .slice(s![.., trim_target..arr.dim().1 - trim_target, ..])
                .to_owned()
        } else {
            arr = arr.as_standard_layout().to_owned()
        };
        Ok(Values(arr))
    }

    pub fn read_all(&mut self) -> Values {
        let arrs: Vec<_> = self
            .data_store
            .iter_chunks()
            .map(|mut data| {
                if let Some(idx) = self.subset.as_ref() {
                    data = Values(data.0.select(Axis(0), idx));
                }
                if let Some(split) = self.split_data {
                    data = data.split(split.1).unwrap();
                }
                data
            })
            .collect();
        let arr_view = arrs.iter().map(|x| x.0.view()).collect::<Vec<_>>();
        let mut data = ndarray::concatenate(Axis(2), &arr_view).unwrap();

        if let Some(trim_target) = self.trim_target {
            data = data
                .slice(s![.., trim_target..data.dim().1 - trim_target, ..])
                .to_owned()
        } else {
            data = data.as_standard_layout().to_owned()
        };

        Values(data)
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
        let seqs = compress_data_zst(seqs);
        std::fs::write(self.location.join("sequence.dat"), seqs)?;
        Ok(())
    }

    pub fn save_data(
        &mut self,
        data: impl IntoParallelIterator<Item = (String, Values)>,
    ) -> Result<()> {
        self.data_store.write_par(data)
    }

    pub fn consolidate(&mut self, chunk_size: usize) -> Result<()> {
        self.data_store.consolidate(chunk_size)
    }
}

#[derive(Debug, Decode, Encode)]
enum DataStoreIndex {
    KeyValue(#[bincode(with_serde)] IndexMap<String, (usize, usize)>),
    Chunked(#[bincode(with_serde)] (IndexMap<String, (usize, usize)>, Vec<(usize, usize)>)), // chunk index, bucket index
}

impl DataStoreIndex {
    fn keys(&self) -> Box<dyn Iterator<Item = &String> + '_> {
        match self {
            DataStoreIndex::KeyValue(index) => Box::new(index.keys()),
            DataStoreIndex::Chunked((keys, _)) => Box::new(keys.keys()),
        }
    }

    /// (offset, size, bucket)
    fn get(&self, key: &str) -> Option<(usize, usize, usize)> {
        match self {
            DataStoreIndex::KeyValue(index) => {
                let (i, n) = index.get(key)?;
                Some((*i, *n, 0))
            }
            DataStoreIndex::Chunked((keys, lengths)) => {
                let (x, j) = keys.get(key)?;
                let (i, n) = lengths[*x];
                Some((i, n, *j))
            }
        }
    }

    fn chunk_lengths(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            DataStoreIndex::KeyValue(index) => Box::new(index.values().map(|(_, size)| *size)),
            DataStoreIndex::Chunked((_, lengths)) => Box::new(lengths.iter().map(|x| x.1)),
        }
    }
}

#[derive(Debug)]
struct DataStore {
    index: DataStoreIndex,
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
            index: DataStoreIndex::KeyValue(index),
            file,
            location: location.as_ref().to_path_buf(),
        };
        store.write_index()?;
        Ok(store)
    }

    fn write_index(&self) -> Result<()> {
        let index_file = self.location.with_extension("index");
        let index_data = bincode::encode_to_vec(&self.index, bincode::config::standard())?;
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
        let index = if index_file.exists() {
            let index_data = std::fs::read(index_file)?;
            if let Ok(i) = bincode::decode_from_slice(&index_data, bincode::config::standard()) {
                i.0
            } else {
                let i: IndexMap<String, (usize, usize)> =
                    bincode::serde::decode_from_slice(&index_data, bincode::config::standard())?.0;
                DataStoreIndex::KeyValue(i)
            }
        } else {
            bail!(
                "Data store index file not found at {}",
                index_file.display()
            )
        };
        Ok(Self {
            index,
            file,
            location: location.as_ref().to_path_buf(),
        })
    }

    fn read_keys(&mut self, keys: &[String]) -> Values {
        let mut chunks = HashMap::new();
        keys.iter().for_each(|key| {
            let (offset, size, _) = self.index.get(key).unwrap();
            if !chunks.contains_key(&offset) {
                self.file
                    .seek(std::io::SeekFrom::Start(offset as u64))
                    .unwrap();
                let mut buffer = vec![0; size];
                self.file.read_exact(&mut buffer).unwrap();
                let data = Values::decode(&buffer).unwrap().0;
                chunks.insert(offset, data);
            }
        });
        let result: Vec<_> = keys
            .iter()
            .map(|key| {
                let (offset, _, bucket) = self.index.get(key).unwrap();
                let c = chunks.get(&offset).unwrap();
                c.slice(s![.., .., bucket..bucket + 1])
            })
            .collect();

        Values(ndarray::concatenate(Axis(2), &result).unwrap())
    }

    fn iter_chunks(&mut self) -> impl IndexedParallelIterator<Item = Values> + '_ {
        self.file.rewind().unwrap();
        let mut buf = Vec::new();
        self.file.read_to_end(&mut buf).unwrap();
        let mut acc = 0;
        let indices: Vec<_> = self
            .index
            .chunk_lengths()
            .map(move |n| {
                let i = (acc, n);
                acc += n;
                i
            })
            .collect();
        indices.into_par_iter().map(move |(i, n)| {
            if let Ok(v) = Values::decode(&buf[i..i + n]) {
                v
            } else {
                let data = decompress_data_zst(&buf[i..i + n]);
                let data: ArrayD<bf16> = bincode::serde::decode_from_slice::<_, Configuration>(
                    &data,
                    Configuration::default(),
                )
                .unwrap()
                .0;
                Values(
                    data.into_dimensionality::<Ix2>()
                        .unwrap()
                        .insert_axis(Axis(2)),
                )
            }
        })
    }

    fn write_par(
        &mut self,
        data: impl IntoParallelIterator<Item = (String, Values)>,
    ) -> Result<()> {
        if let DataStoreIndex::KeyValue(index) = &mut self.index {
            let mut offset = self.file.seek(std::io::SeekFrom::End(0))? as usize;
            let data: Vec<_> = data
                .into_par_iter()
                .map(|(key, values)| (key, values.encode().unwrap()))
                .collect();
            data.into_iter().for_each(|(key, buffer)| {
                let size = buffer.len();
                self.file.write_all(&buffer).unwrap();
                index.insert(key, (offset, size));
                offset += size;
            });
            self.file.flush()?;
            self.write_index()?;
            Ok(())
        } else {
            bail!("Cannot write to a chunked data store. Use a key-value store instead.");
        }
    }

    /// Consolidate the data store by combining all entries into a single array for
    /// faster loading and reduced file size.
    fn consolidate(&mut self, chunk_size: usize) -> Result<()> {
        let keys: IndexMap<_, _> = self
            .index
            .keys()
            .cloned()
            .enumerate()
            .map(|(i, x)| (x, (i / chunk_size, i % chunk_size)))
            .collect();
        let arrs: Vec<_> = self
            .iter_chunks()
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk = chunk.iter().map(|x| x.0.view()).collect::<Vec<_>>();
                let arr = ndarray::concatenate(Axis(2), &chunk).unwrap();
                Values(arr).encode().unwrap()
            })
            .collect();

        self.file.set_len(0)?;
        self.file.rewind()?;

        let mut acc = 0;
        let index: Vec<_> = arrs
            .into_iter()
            .map(|data| {
                let r = (acc, data.len());
                self.file.write_all(&data).unwrap();
                acc += data.len();
                r
            })
            .collect();
        let index = DataStoreIndex::Chunked((keys, index));
        let index_file = self.location.with_extension("index");
        let index_data = bincode::encode_to_vec::<_, Configuration>(&index, Default::default())?;
        std::fs::write(index_file, index_data)?;
        self.file.flush()?;
        self.index = index;

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

fn compress_data_zst(data: Vec<u8>) -> Vec<u8> {
    zstd::bulk::Compressor::new(9)
        .unwrap()
        .compress(&data)
        .unwrap()
}

fn decompress_data_zst(buffer: &[u8]) -> Vec<u8> {
    let mut decoder = zstd::Decoder::new(buffer).unwrap();
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();
    decompressed_data
}