pub mod genome;

use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use std::{
    fs::File, io::{Read, Seek, Write}, path::{Path, PathBuf}
};

use anyhow::{Context, Result};

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
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .append(false)
            .create(true)
            .open(&data_file)
            .with_context(|| {
                format!(
                    "Failed to create data file at {}",
                    data_file.as_ref().display()
                )
            })?;

        let sizes = data
            .map(|item| {
                let mut buffer =
                    bincode::serde::encode_to_vec(item, bincode::config::standard()).unwrap();
                buffer = compress_data_zst(buffer, 5);
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

    pub fn iter(&self) -> DataLoaderIter {
        let file = File::open(&self.path).expect("Failed to open data file");
        DataLoaderIter { file, sizes: self.sizes.clone(), pos: 0 }
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
}

#[pyclass]
pub struct DataLoaderIter {
    file: File,
    sizes: Vec<usize>,
    pos: usize,
}

impl Iterator for DataLoaderIter {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.sizes.len() {
            return None;
        }

        let size = self.sizes[self.pos];
        let mut buffer = vec![0; size];
        self.file.read_exact(&mut buffer).unwrap();

        buffer = decompress_data_zst(&buffer);
        self.pos += 1;
        Some(buffer)
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
