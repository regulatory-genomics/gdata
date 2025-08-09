use anyhow::{bail, Context, Result};
use bed_utils::bed::GenomicRange;
use bincode::config::Configuration;
use bincode::{Decode, Encode};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array1, Array2, Array3, Axis};
use numpy::PyArray2;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::io::{Read, Seek, Write};
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Decode, Encode, PartialEq)]
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

    /// Aggregate the values along the sequence axis (axis 1).
    fn aggregate_by_length(self, size: usize) -> Self {
        let (d, h, w) = self.0.dim();
        if h % size != 0 {
            panic!(
                "Cannot aggregate values of length {} by size {}: length is not a multiple of size",
                h, size
            );
        }
        let num_chunks = h / size;
        let data = self
            .0
            .into_shape_with_order((d, num_chunks, size, w))
            .unwrap()
            .mapv(|x| x.to_f64())
            .mean_axis(Axis(2))
            .unwrap()
            .mapv(|x| bf16::from_f64(x));
        Values(data)
    }

    fn decode(buffer: &[u8]) -> Result<Self> {
        let data = decompress_data_zst(buffer);
        Ok(bincode::decode_from_slice::<_, Configuration>(&data, Configuration::default())?.0)
    }

    fn encode(self, compression: u8) -> Result<Vec<u8>> {
        let data = bincode::encode_to_vec::<_, Configuration>(self, Configuration::default())?;
        let data = compress_data_zst(data, compression);
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
    split_data: Option<(usize, usize)>, // Optional size for splitting data, 1st is for sequences, 2nd for values
    aggregation: Option<usize>, // Optional aggregation size for the data store
    data_store: DataStore,              // Data store
    subset: Option<Vec<usize>>,         // Optional indices representing a subset of the data
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
            aggregation: None,
            data_store: DataStore::create(location.as_ref().join("data"))?,
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
            aggregation: None,
            data_store: DataStore::open(store_path, writable)?,
            subset: None,
        })
    }

    /// Set the output trimming target for the data store.
    pub fn set_trim_target(&mut self, trim_target: usize) {
        self.trim_target = Some(trim_target);
    }

    /// Set the split data size for the data store.
    pub fn set_split_data(&mut self, split_data: (usize, usize)) {
        self.split_data = Some(split_data);
    }

    /// Set the aggregation size for the data store.
    pub fn set_aggregation(&mut self, aggregation: usize) {
        self.aggregation = Some(aggregation);
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

    pub fn read(&mut self, idx: &[usize]) -> Result<Values> {
        let mut data = self.data_store.read(idx, self.aggregation);
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

    pub fn read_keys(&mut self, keys: &[String]) -> Result<Values> {
        let mut data = self.data_store.read_keys(keys, self.aggregation);
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
            .iter_chunks(self.aggregation)
            .flat_map_iter(|chunk| {
                chunk.into_iter().map(|mut data| {
                    if let Some(idx) = self.subset.as_ref() {
                        data = Values(data.0.select(Axis(0), idx));
                    }
                    if let Some(split) = self.split_data {
                        data = data.split(split.1).unwrap();
                    }
                    data
                })
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

    pub fn save_seqs(&self, seqs: Vec<Vec<u8>>, compression: u8) -> Result<()> {
        let shape = (seqs.len(), seqs[0].len());
        let seqs: Result<Vec<u8>> = seqs
            .into_iter()
            .flatten()
            .map(|x| encode_nucleotide(x))
            .collect();
        let seqs = Sequences(Array2::from_shape_vec(shape, seqs?)?);
        let seqs = bincode::encode_to_vec::<_, Configuration>(seqs, Default::default())?;
        let seqs = compress_data_zst(seqs, compression);
        std::fs::write(self.location.join("sequence.dat"), seqs)?;
        Ok(())
    }

    pub fn save_data(
        &mut self,
        data: impl IntoParallelIterator<Item = (String, Values)>,
    ) -> Result<()> {
        self.data_store.write_par(data, 9)
    }

    pub fn compress(&mut self, lvl: u8) -> Result<()> {
        // compress sequences
        let seq_file = self.location.join("sequence.dat");
        let seqs = Sequences::decode(&std::fs::read(&seq_file)?)?;
        let seqs = bincode::encode_to_vec(seqs, bincode::config::standard())?; 
        let seqs = compress_data_zst(seqs, lvl);
        std::fs::write(seq_file, seqs)?;

        // compress data store
        let data: Vec<_> = self.data_store.iter_chunks(None).flatten().collect();
        let data: Vec<_> = self.data_store.index.keys().cloned().zip(data.into_iter()).collect();
        self.data_store.clear()?;
        self.data_store.write_par(data, lvl)?;
        Ok(())
    }
}

#[derive(Debug, Decode, Encode)]
struct DataStoreIndex(#[bincode(with_serde)] IndexMap<String, (usize, usize)>);

impl DataStoreIndex {
    fn keys(&self) -> impl Iterator<Item = &String> + '_ {
        self.0.keys()
    }

    /// (offset, size)
    fn get(&self, key: &str) -> Option<(usize, usize)> {
        let (i, n) = self.0.get(key)?;
        Some((*i, *n))
    }

    fn get_index(&self, idx: usize) -> Option<(usize, usize)> {
        self.0.get_index(idx).map(|(_, &(i, n))| (i, n))
    }
}

#[derive(Debug)]
struct DataStore {
    index: DataStoreIndex,
    file: std::fs::File,
}

impl DataStore {
    fn clear(&mut self) -> Result<()> {
        self.index = DataStoreIndex(IndexMap::new());
        self.file.set_len(0)?;
        self.file.rewind()?;
        self.write_index()?;
        Ok(())
    }

    fn create(location: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .append(false)
            .create(true)
            .open(&location)
            .with_context(|| {
                format!(
                    "Failed to create data file at {}",
                    location.as_ref().display()
                )
            })?;
        let index = DataStoreIndex(IndexMap::new());

        let mut store = Self {
            index,
            file,
        };
        store.write_index()?;
        Ok(store)
    }

    fn write_index(&mut self) -> Result<()> {
        let pos = self.file.seek(std::io::SeekFrom::End(0))?;
        let index_data = bincode::encode_to_vec(&self.index, bincode::config::standard())?;
        let n_bytes = index_data.len();
        self.file.write_all(&index_data)?;
        self.file.write_all(&pos.to_le_bytes())?;
        self.file.write_all(&(n_bytes as u32).to_le_bytes())?;
        Ok(())
    }

    fn open(location: impl AsRef<Path>, writable: bool) -> Result<Self> {
        let mut opt = std::fs::OpenOptions::new();
        let opt = opt.create(false).read(true).append(false);
        let opt = if writable {
            opt.write(true)
        } else {
            opt.write(false)
        };
        let mut file = opt.open(&location).with_context(|| {
            format!(
                "Failed to open data file at {}",
                location.as_ref().display()
            )
        })?;

        let index = read_index_from_file(&mut file)?;
        Ok(Self {
            index,
            file,
        })
    }

    /// Find the position in the file where the next chunk should be inserted.
    fn seek_insert_loc(&mut self) -> Result<u64> {
        self.file.seek(std::io::SeekFrom::End(-12))?;
        let mut start = [0; 8];
        self.file.read_exact(&mut start)?;
        let start = u64::from_le_bytes(start);

        Ok(self.file.seek(std::io::SeekFrom::Start(start))?)
    }

    fn read(&mut self, idx: &[usize], aggregation: Option<usize>) -> Values {
        let result = idx
            .iter()
            .map(|i| {
                let (offset, size) = self.index.get_index(*i).unwrap();
                self.file
                    .seek(std::io::SeekFrom::Start(offset as u64))
                    .unwrap();
                let mut buffer = vec![0; size];
                self.file.read_exact(&mut buffer).unwrap();
                let mut data = Values::decode(&buffer).unwrap();
                if let Some(aggregation) = aggregation {
                    data = data.aggregate_by_length(aggregation);
                }
                data.0
            })
            .collect::<Vec<_>>();
        let result = result.iter().map(|x| x.view()).collect::<Vec<_>>();
        Values(ndarray::concatenate(Axis(2), &result).unwrap())
    }

    fn read_keys(&mut self, keys: &[String], aggregation: Option<usize>) -> Values {
        let result = keys
            .iter()
            .map(|key| {
                let (offset, size) = self.index.get(key).unwrap();
                self.file
                    .seek(std::io::SeekFrom::Start(offset as u64))
                    .unwrap();
                let mut buffer = vec![0; size];
                self.file.read_exact(&mut buffer).unwrap();
                let mut data = Values::decode(&buffer).unwrap();
                if let Some(aggregation) = aggregation {
                    data = data.aggregate_by_length(aggregation);
                }
                data.0
            })
            .collect::<Vec<_>>();
        let result = result.iter().map(|x| x.view()).collect::<Vec<_>>();
        Values(ndarray::concatenate(Axis(2), &result).unwrap())
    }

    fn iter_chunks(&mut self, aggregation: Option<usize>) -> impl ParallelIterator<Item = Vec<Values>> + '_ {
        self.file.rewind().unwrap();
        let mut buf = Vec::new();
        self.file.read_to_end(&mut buf).unwrap();
        let chunks: Vec<Vec<_>> = self
            .index
            .0
            .values()
            .chunks(256)
            .into_iter()
            .map(|x| x.collect())
            .collect();

        chunks.into_par_iter().map(move |c| {
            c.into_iter()
                .map(|&(i, n)| {
                    let mut data = Values::decode(&buf[i..i + n]).unwrap();
                    if let Some(aggregation) = aggregation {
                        data = data.aggregate_by_length(aggregation);
                    }
                    data
                })
                .collect::<Vec<_>>()
        })
    }

    fn write_par(
        &mut self,
        data: impl IntoParallelIterator<Item = (String, Values)>,
        compression: u8,
    ) -> Result<()> {
        let data: Vec<_> = data
            .into_par_iter()
            .map(|(key, values)| (key, values.encode(compression).unwrap()))
            .collect();
        let mut offset = self.seek_insert_loc()? as usize;
        data.into_iter().for_each(|(key, buffer)| {
            let size = buffer.len();
            self.file.write_all(&buffer).unwrap();
            self.index.0.insert(key, (offset, size));
            offset += size;
        });

        self.file.set_len(offset as u64).unwrap();
        self.write_index()?;
        self.file.flush()?;
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

fn compress_data_zst(data: Vec<u8>, lvl: u8) -> Vec<u8> {
    zstd::bulk::Compressor::new(lvl as i32)
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

fn read_index_from_file(file: &mut std::fs::File) -> Result<DataStoreIndex> {
    file.seek(std::io::SeekFrom::End(-12))?;
    let mut start = [0; 8];
    file.read_exact(&mut start)?;
    let start = u64::from_le_bytes(start);

    let mut len = [0; 4];
    file.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len);

    file.seek(std::io::SeekFrom::Start(start))?;
    let mut index_data = vec![0; len as usize];
    file.read_exact(&mut index_data)?;
    let index: DataStoreIndex =
        bincode::decode_from_slice(&index_data, bincode::config::standard())?.0;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_arr_aggregation() {
        let arr = array![
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]
        ];
        assert_eq!(arr.dim(), (2, 6, 1));

        let values = Values(arr.mapv(bf16::from_f64));
        let aggregated = values.aggregate_by_length(2).0.mapv(|x| x.to_f64());
        assert_eq!(aggregated.dim(), (2, 3, 1));
        assert_eq!(
            aggregated,
            array![[[1.5], [3.5], [5.5]], [[7.5], [9.5], [11.5]]]
        );
    }

    #[test]
    fn test_datastore() {
        let temp_dir = tempfile::tempdir().unwrap();
        let location = temp_dir.path().join("datastore");
        let mut store = DataStore::create(&location).unwrap();

        let values1 =
            Values(array![[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]].mapv(bf16::from_f64));
        let values2 =
            Values(array![[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]].mapv(bf16::from_f64));

        store
            .write_par(vec![
                ("key1".to_string(), values1),
                ("key2".to_string(), values2),
            ], 9)
            .unwrap();

        let read_values = store
            .read_keys(&["key1".to_string(), "key2".to_string()], None)
            .0
            .as_standard_layout()
            .to_owned();
        assert_eq!(read_values.dim(), (2, 3, 2));
        assert_eq!(
            read_values,
            array![
                [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
                [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]]
            ]
            .mapv(bf16::from_f64)
        );
    }
}
