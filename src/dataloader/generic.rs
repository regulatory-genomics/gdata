use ndarray::{Array, ArrayD, Dimension};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::{prelude::*, types::PyType};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use rayon::str;
use serde::Serialize;
use std::os::unix::fs::FileExt;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::{
    collections::VecDeque,
    fs::File,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

/// A generic data loader that serializes data items to a file and allows iteration over them.
pub struct DataLoader {
    path: PathBuf,
    offset_and_size: Vec<(usize, usize)>,
}

impl DataLoader {
    /// Create a new DataLoader instance from an iterator of data items.
    pub fn new<I, D>(data: I, data_file: impl AsRef<Path>) -> Result<Self>
    where
        I: Iterator<Item = D>,
        D: serde::Serialize,
    {
        let mut file = File::create(&data_file).with_context(|| {
            format!(
                "Failed to create data file at {}",
                data_file.as_ref().display()
            )
        })?;

        let sizes: Vec<usize> = data
            .map(|item| {
                let mut buffer =
                    bincode::serde::encode_to_vec(item, bincode::config::standard()).unwrap();
                buffer = compress_data_zst(buffer, 9);
                let size = buffer.len();
                file.write_all(&buffer).unwrap();
                size
            })
            .collect();

        let pos = file.seek(std::io::SeekFrom::End(0))?;
        let index_data = bincode::encode_to_vec(&sizes, bincode::config::standard())?;
        let n_bytes = index_data.len();
        file.write_all(&index_data)?;
        file.write_all(&pos.to_le_bytes())?;
        file.write_all(&(n_bytes as u32).to_le_bytes())?;

        Ok(Self {
            path: data_file.as_ref().to_path_buf(),
            offset_and_size: sizes
                .into_iter()
                .scan(0, |offset, size| {
                    let current_offset = *offset;
                    *offset += size;
                    Some((current_offset, size))
                })
                .collect(),
        })
    }

    pub fn open(data_file: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(&data_file).with_context(|| {
            format!(
                "Failed to open data file at {}",
                data_file.as_ref().display()
            )
        })?;
        let file_size = file.metadata()?.len();

        if file_size < 12 {
            return Err(anyhow::anyhow!(
                "Data file is too small to contain index information"
            ));
        }

        file.seek(std::io::SeekFrom::End(-12))?;

        let mut pos_bytes = [0u8; 8];
        let mut n_bytes_bytes = [0u8; 4];
        file.read_exact(&mut pos_bytes)?;
        file.read_exact(&mut n_bytes_bytes)?;

        let pos = u64::from_le_bytes(pos_bytes);
        let n_bytes = u32::from_le_bytes(n_bytes_bytes) as usize;

        if pos + n_bytes as u64 + 12 != file_size {
            return Err(anyhow::anyhow!(
                "Index position and size do not match file size"
            ));
        }

        file.seek(std::io::SeekFrom::Start(pos))?;

        let mut index_data = vec![0u8; n_bytes];
        file.read_exact(&mut index_data)?;

        let sizes: Vec<usize> =
            bincode::decode_from_slice(&index_data, bincode::config::standard())?.0;
        Ok(Self {
            path: data_file.as_ref().to_path_buf(),
            offset_and_size: sizes
                .into_iter()
                .scan(0, |offset, size| {
                    let current_offset = *offset;
                    *offset += size;
                    Some((current_offset, size))
                })
                .collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.offset_and_size.len()
    }

    fn iter<T: serde::de::DeserializeOwned + Send>(&self) -> ParallelLoader<DataLoaderIter<T>, T> {
        let file = File::open(&self.path).expect("Failed to open data file");
        let num_works = 16;
        let iters = split_into_n(&self.offset_and_size, num_works)
            .into_iter()
            .map(|chunk| DataLoaderIter {
                file: file.try_clone().expect("Failed to clone file handle"),
                offset_and_size: chunk,
                pos: 0,
                _phantom: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();
        ParallelLoader::new(iters)
    }
}

struct DataLoaderIter<T> {
    file: File,
    offset_and_size: Vec<(usize, usize)>,
    pos: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: serde::de::DeserializeOwned> Iterator for DataLoaderIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let (offset, size) = self.offset_and_size.get(self.pos)?;
        let mut buffer = vec![0; *size];
        self.file
            .read_exact_at(&mut buffer, *offset as u64)
            .expect("read failed");
        let decompressed_data = decompress_data_zst(&buffer);
        let item: T =
            bincode::serde::decode_from_slice(&decompressed_data, bincode::config::standard())
                .expect("deserialization failed")
                .0;
        self.pos += 1;
        Some(item)
    }
}

/** A array data loader that serializes numpy arrays to a file and allows iteration over them.

    Parameters
    ----------
    data : Iterator[tuple[np.ndarray, np.ndarray]]
        An iterator that yields data items to be serialized and stored.
    data_file : str
        The path to the file where the serialized data will be stored.
*/
#[pyclass]
pub(crate) struct PyArrayDataLoader {
    inner: DataLoader,
    batch_size: usize,
}

struct PyArrayWrapper<'py, T>(Bound<'py, PyArrayDyn<T>>);

impl<T: Serialize + numpy::Element> Serialize for PyArrayWrapper<'_, T> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let array = self.0.readonly();
        array.as_array().serialize(serializer)
    }
}

#[pymethods]
impl PyArrayDataLoader {
    #[new]
    #[pyo3(
        signature = (data, data_file, batch_size=32),
        text_signature = "($self, data, data_file, batch_size=32)"
    )]
    pub fn new(data: Bound<'_, PyAny>, data_file: PathBuf, batch_size: usize) -> Result<Self> {
        let iter = data.try_iter()?.map(|item| {
            let (arr1, arr2): (Bound<'_, PyArrayDyn<f32>>, Bound<'_, PyArrayDyn<f32>>) =
                item.unwrap().extract().unwrap();
            (PyArrayWrapper(arr1), PyArrayWrapper(arr2))
        });
        let dataloader = DataLoader::new(
            iter, //PrefethIterator::new(iter, 32),
            data_file,
        )?;
        Ok(Self {
            inner: dataloader,
            batch_size,
        })
    }

    /** Open an existing DataLoader from a data file.

        Parameters
        ----------
        data_file
            The path to the file where the serialized data is stored.

        Returns
        -------
        DataLoader
            An instance of DataLoader initialized with the data from the specified file.
    */
    #[classmethod]
    #[pyo3(
        signature = (data_file, batch_size=32),
        text_signature = "(data_file, batch_size=32)"
    )]
    fn open(_cls: &Bound<'_, PyType>, data_file: PathBuf, batch_size: usize) -> Result<Self> {
        Ok(Self {
            inner: DataLoader::open(data_file)?,
            batch_size,
        })
    }

    fn __len__(&self) -> usize {
        let n = self.inner.len() / self.batch_size;
        if self.inner.len() % self.batch_size == 0 {
            n
        } else {
            n + 1
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyDataLoaderIter {
        let iter = ReBatch::new(slf.inner.iter(), slf.batch_size);
        PyDataLoaderIter(PrefethIterator::new(iter, 32))
    }
}

#[pyclass]
struct PyDataLoaderIter(PrefethIterator<(ArrayD<f32>, ArrayD<f32>)>);

#[pymethods]
impl PyDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<(Bound<'a, PyArrayDyn<f32>>, Bound<'a, PyArrayDyn<f32>>)> {
        let (input, target) = slf.0.next()?;
        Some((
            PyArrayDyn::from_owned_array(py, input),
            PyArrayDyn::from_owned_array(py, target),
        ))
    }
}

pub struct ParallelLoader<L, T> {
    loaders: Vec<L>,
    buffer: VecDeque<T>,
}

impl<L, T> Iterator for ParallelLoader<L, T>
where
    T: Send,
    L: Iterator<Item = T> + Send,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.load();
        self.buffer.pop_front()
    }
}

impl<L, T> ParallelLoader<L, T>
where
    T: Send,
    L: Iterator<Item = T> + Send,
    //for<'a> &'a Vec<L>: IntoParallelIterator<Item = L>,
{
    pub fn new(loaders: Vec<L>) -> Self {
        Self {
            loaders,
            buffer: VecDeque::new(),
        }
    }

    /// Fill the internal buffer if it is empty.
    pub fn load(&mut self) {
        if self.buffer.is_empty() {
            self.buffer = self
                .loaders
                .par_iter_mut()
                .flat_map(|loader| loader.next())
                .collect();
        }
    }
}

/// PrefetchIterator allows for prefetching items from an iterator into a buffer.
pub struct PrefethIterator<T>(Arc<Mutex<Receiver<T>>>);

impl<T: Send + 'static> PrefethIterator<T> {
    pub fn new<I>(iter: I, buffer_size: usize) -> Self
    where
        I: IntoIterator<Item = T> + Send + 'static,
    {
        let (sender, receiver) = sync_channel(buffer_size);
        std::thread::spawn(move || {
            for item in iter {
                if sender.send(item).is_err() {
                    // If the receiver is dropped, we stop sending more items.
                    break;
                }
            }
        });
        Self(Arc::new(Mutex::new(receiver)))
    }
}

impl<T: Send + 'static> Iterator for PrefethIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.lock().unwrap().recv().ok()
    }
}

/// Rebatch an iterator of arrays into a specified batch size.
pub(crate) struct ReBatch<T1, T2, D1, D2, I> {
    batch_size: usize,
    input_buffer: ArrayBuffer<T1, D1>,
    target_buffer: ArrayBuffer<T2, D2>,
    chunks: I,
}

impl<T1: Clone, T2: Clone, D1: Dimension, D2: Dimension, I> ReBatch<T1, T2, D1, D2, I> {
    pub fn new(iter: I, batch_size: usize) -> Self {
        Self {
            batch_size,
            input_buffer: ArrayBuffer::new(),
            target_buffer: ArrayBuffer::new(),
            chunks: iter,
        }
    }
}

impl<T1, T2, D1, D2, I> Iterator for ReBatch<T1, T2, D1, D2, I>
where
    T1: Clone,
    T2: Clone,
    D1: Dimension,
    D2: Dimension,
    I: Iterator<Item = (Array<T1, D1>, Array<T2, D2>)>,
{
    type Item = (Array<T1, D1>, Array<T2, D2>);

    fn next(&mut self) -> Option<Self::Item> {
        let n_buffer = self.input_buffer.num_rows();
        if n_buffer < self.batch_size {
            if let Some((inputs, targets)) = self.chunks.next() {
                self.input_buffer.add(inputs);
                self.target_buffer.add(targets);
                self.next()
            } else {
                if n_buffer == 0 {
                    None
                } else {
                    let inputs = self.input_buffer.take_all();
                    let targets = self.target_buffer.take_all();
                    Some((inputs, targets))
                }
            }
        } else {
            let inputs = self.input_buffer.take(self.batch_size).unwrap();
            let targets = self.target_buffer.take(self.batch_size).unwrap();
            Some((inputs, targets))
        }
    }
}

/// An ArrayBuffer stores arrays as a list internally. It supports retrieving items in a batch-wise manner,
/// e.g., rebatching the data into arrays with a specified number of records.
struct ArrayBuffer<T, D> {
    data: VecDeque<Vec<T>>,
    pos: usize,
    shape: Vec<usize>,
    stride: usize,
    _phantom: std::marker::PhantomData<D>,
}

impl<T: Clone, D: Dimension> ArrayBuffer<T, D> {
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
            pos: 0,
            shape: Vec::new(),
            stride: 1,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the total number of remaining items/rows in the buffer.
    fn num_rows(&self) -> usize {
        let n = self.data.iter().map(|d| d.len()).sum::<usize>();
        (n - self.pos) / self.stride
    }

    /// Add an item to the buffer. The first item determines the shape and stride of the buffer.
    pub fn add(&mut self, item: Array<T, D>) {
        if self.shape.is_empty() {
            self.shape = item.shape()[1..].to_vec();
            self.stride = self.shape.iter().product();
        }
        assert!(
            item.is_standard_layout(),
            "Buffer only supports standard layout arrays"
        );
        let (data, offset) = item.into_raw_vec_and_offset();
        assert!(
            offset.unwrap_or(0) == 0,
            "Buffer does not support non-zero offset"
        );
        self.data.push_back(data);
    }

    /// Take a specified number of records from the buffer and return them as an `ArrayD`.
    pub fn take(&mut self, n_record: usize) -> Option<Array<T, D>> {
        if self.data.is_empty() {
            return None;
        }

        let n = self.stride * n_record;
        let mut result = Vec::with_capacity(n);
        while result.len() < n && !self.data.is_empty() {
            let remaining = self.data[0].len() - self.pos;
            if remaining == 0 {
                self.data.pop_front();
                self.pos = 0;
                continue;
            }

            let take_count = remaining.min(n - result.len());
            result.extend_from_slice(&self.data[0][self.pos..self.pos + take_count]);
            self.pos += take_count;

            if self.pos >= self.data[0].len() {
                self.pos = 0;
                self.data.pop_front();
            }
        }

        if result.is_empty() {
            None
        } else {
            let shape = std::iter::once(result.len() / self.stride)
                .chain(self.shape.iter().cloned())
                .collect::<Vec<_>>();
            let arr = ArrayD::from_shape_vec(shape, result)
                .unwrap()
                .into_dimensionality::<D>()
                .unwrap();
            Some(arr)
        }
    }

    /// Take all remaining records from the buffer and return them as an `ArrayD`.
    pub fn take_all(&mut self) -> Array<T, D> {
        let n = self.num_rows();
        let data: Vec<_> = self.data.drain(..).flatten().skip(self.pos).collect();
        let shape = std::iter::once(n)
            .chain(self.shape.iter().cloned())
            .collect::<Vec<_>>();
        self.pos = 0;
        ArrayD::from_shape_vec(shape, data)
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap()
    }
}

/// Compress data using Zstandard with the specified compression level.
pub(crate) fn compress_data_zst(data: Vec<u8>, lvl: u8) -> Vec<u8> {
    zstd::bulk::Compressor::new(lvl as i32)
        .unwrap()
        .compress(&data)
        .unwrap()
}

/// Decompress data using Zstandard.
pub(crate) fn decompress_data_zst(buffer: &[u8]) -> Vec<u8> {
    let mut decoder = zstd::Decoder::new(buffer).unwrap();
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();
    decompressed_data
}

pub(crate) fn split_into_n<T: Clone>(values: &[T], n: usize) -> Vec<Vec<T>> {
    let mut result = vec![Vec::new(); n];
    values.iter().enumerate().for_each(|(i, v)| {
        result[i % n].push(v.clone());
    });
    result
}
