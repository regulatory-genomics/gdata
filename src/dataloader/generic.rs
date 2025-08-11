use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::{prelude::*, types::PyType};
use std::{
    fs::File, io::{Read, Seek, Write}, path::{Path, PathBuf}
};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

/** A generic data loader that serializes data items to a file and allows iteration over them.
    
    Parameters
    ----------
    data : Iterator
        An iterator that yields data items to be serialized and stored.
    data_file : str
        The path to the file where the serialized data will be stored.
*/
#[pyclass]
pub struct DataLoader {
    path: PathBuf,
    sizes: Vec<usize>,
}

impl DataLoader {
    /// Create a new DataLoader instance from an iterator of data items.
    pub fn new<I, D>(data: I, data_file: impl AsRef<Path>) -> Result<Self>
    where
        I: Iterator<Item = D>,
        D: serde::Serialize,
    {
        let mut file = File::create(&data_file)
            .with_context(|| format!("Failed to create data file at {}", data_file.as_ref().display()))?;

        let sizes = data
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

        Ok(Self { path: data_file.as_ref().to_path_buf(), sizes })
    }

    pub fn open(data_file: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(&data_file)
            .with_context(|| format!("Failed to open data file at {}", data_file.as_ref().display()))?;
        let file_size = file.metadata()?.len();

        if file_size < 12 {
            return Err(anyhow::anyhow!("Data file is too small to contain index information"));
        }

        file.seek(std::io::SeekFrom::End(-12))?;

        let mut pos_bytes = [0u8; 8];
        let mut n_bytes_bytes = [0u8; 4];
        file.read_exact(&mut pos_bytes)?;
        file.read_exact(&mut n_bytes_bytes)?;

        let pos = u64::from_le_bytes(pos_bytes);
        let n_bytes = u32::from_le_bytes(n_bytes_bytes) as usize;

        if pos + n_bytes as u64 + 12 != file_size {
            return Err(anyhow::anyhow!("Index position and size do not match file size"));
        }

        file.seek(std::io::SeekFrom::Start(pos))?;

        let mut index_data = vec![0u8; n_bytes];
        file.read_exact(&mut index_data)?;

        let sizes = bincode::decode_from_slice(&index_data, bincode::config::standard())?.0;

        Ok(Self { path: data_file.as_ref().to_path_buf(), sizes })
    }

    fn iter<T>(&self) -> DataLoaderIter<T> {
        let file = File::open(&self.path).expect("Failed to open data file");
        DataLoaderIter { file, sizes: self.sizes.clone(), pos: 0, _phantom: std::marker::PhantomData }
    }
}

#[pymethods]
impl DataLoader {
    #[new]
    #[pyo3(
        signature = (data, data_file),
        text_signature = "($self, data, data_file)"
    )]
    pub fn new_py(data: Bound<'_, PyAny>, data_file: PathBuf) -> Result<Self> {
        let iter = data.try_iter()?.map(|item| {
            let item = item.unwrap();
            let (input, target): (Bound<'_, PyArrayDyn<f32>>, Bound<'_, PyArrayDyn<f32>>) = item.extract().unwrap();
            (input.to_owned_array(), target.to_owned_array())
        });
        Self::new(iter, data_file)
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
        name = "open",
        signature = (data_file),
        text_signature = "(data_file)"
    )]
    fn open_py(_cls: &Bound<'_, PyType>, data_file: PathBuf) -> Result<Self> {
        Self::open(data_file)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyDataLoaderIter {
        let iter = slf.iter();
        PyDataLoaderIter(PrefethIterator::new(iter, 32))
    }
}

struct DataLoaderIter<T> {
    file: File,
    sizes: Vec<usize>,
    pos: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: serde::de::DeserializeOwned> Iterator for DataLoaderIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.sizes.len() {
            return None;
        }

        let size = self.sizes[self.pos];
        let mut buffer = vec![0; size];
        self.file.read_exact(&mut buffer).unwrap();

        buffer = decompress_data_zst(&buffer);
        let item = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap().0;
        self.pos += 1;
        Some(item)
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
        Some((PyArrayDyn::from_owned_array(py, input), PyArrayDyn::from_owned_array(py, target)))
    }
}


pub struct PrefethIterator<T>(Arc<Mutex<Receiver<T>>>);

impl<T: Send + 'static> PrefethIterator<T> {
    pub fn new<I>(iter: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
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

pub(crate) fn compress_data_zst(data: Vec<u8>, lvl: u8) -> Vec<u8> {
    zstd::bulk::Compressor::new(lvl as i32)
        .unwrap()
        .compress(&data)
        .unwrap()
}

pub(crate) fn decompress_data_zst(buffer: &[u8]) -> Vec<u8> {
    let mut decoder = zstd::Decoder::new(buffer).unwrap();
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();
    decompressed_data
}