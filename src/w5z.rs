use anyhow::Result;
use bincode::config::Configuration;
use hdf5::File;
use ndarray::{Array1, ArrayView1};
use numpy::PyArray1;
use pyo3::prelude::*;
use std::path::PathBuf;

/** W5Z object is a zfp compressed basepair resolution data format for representing genome-wide
 * coverage data, such as ChIP-seq, ATAC-seq, and other genomic signals.
*/
#[pyclass]
#[repr(transparent)]
pub struct W5Z {
    inner: File,
}

#[pymethods]
impl W5Z {
    #[new]
    #[pyo3(
        signature = (filename),
        text_signature = "($self, filename)"
    )]
    fn new(filename: PathBuf) -> Result<Self> {
        let inner = File::open(filename)?;
        Ok(Self { inner })
    }

    fn keys(&self) -> Result<Vec<String>> {
        let group = self.inner.group("/")?;
        let keys = group
            .datasets()?
            .into_iter()
            .map(|d| d.name().trim_start_matches('/').to_string())
            .collect();
        Ok(keys)
    }

    fn __getitem__<'py>(&'py self, py: Python<'py>, key: &str) -> Result<Bound<'py, PyArray1<f32>>> {
        let group = self.inner.group("/")?;
        let dataset = group.dataset(key)?;
        let size = dataset.attr("zstd_size")?.read_scalar::<u64>()?;
        let zfp = dataset.attr("zfp")?.read_scalar::<bool>()?;
        let arr = decode_z(&dataset.read_1d()?, zfp, size as usize)?;
        Ok(PyArray1::from_owned_array(py, arr))
    }

    fn verify(&self, py: Python) -> Result<()> {
        let chromosomes: Vec<_> = self.keys()?;
        let mut s = 0.0f64;
        for chrom in chromosomes {
            let data = self.__getitem__(py, &chrom)?;
            for x in data.try_iter()? {
                s += x?.extract::<f64>()?;
            }
        }
        let real_s = self.inner.attr("sum")?.read_scalar::<f64>()?;
        println!("Sum of all values: {}; Delta with ground truth: {}", s, (s - real_s).abs());
        Ok(())
    }

    fn close(&self) -> Result<()> {
        self.inner.clone().close()?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __str__(&self) -> String {
        format!("W5Z object with keys: {}", self.keys().unwrap().join(", "))
    }
}

pub struct Codec {
    pub zfp: Option<f64>,
    pub zstd_src_size: u64,
}

/// Encode the data using ZFP compression with a specified precision.
pub fn encode_z(data: &[f32], zfp: Option<bool>, tolerance: f64, compression_level: u8) -> Result<(Codec, Vec<u8>)> {
    match zfp {
        Some(true) => enc_both(data, tolerance, compression_level),
        Some(false) => enc_zstd(data, compression_level),
        None => {
            let (c1, d1) = enc_both(data, tolerance, compression_level)?;
            let (c2, d2) = enc_zstd(data, compression_level)?;
            if d1.len() < d2.len() {
                Ok((c1, d1))
            } else {
                Ok((c2, d2))
            }
        }
    }
}

pub fn decode_z(data: &Array1<u8>, zfp: bool, size: usize) -> Result<Array1<f32>> {
    let data = zstd::bulk::decompress(data.as_slice().unwrap(), size)
        .map_err(|e| anyhow::anyhow!("Failed to decompress zstd data: {}", e))?;
    let decoded: Array1<f32> = if zfp {
        numcodecs_zfp::decompress(&data)?
            .as_typed::<f32>()
            .unwrap()
            .to_owned()
            .into_dimensionality()?
    } else {
        Array1::from_vec(bincode::decode_from_slice::<_, Configuration>(&data, Configuration::default())?.0)
    };
    Ok(decoded)
}

fn enc_both(data: &[f32], tolerance: f64, compression_level: u8) -> Result<(Codec, Vec<u8>)> {
    let data = enc_zfp(data, tolerance)?;
    let size = data.len();
    let mut zstd = zstd::bulk::Compressor::new(compression_level as i32)?;
    zstd.multithread(256)?;
    let zstd_data = zstd.compress(&data)?;
    let codec = Codec {
        zfp: Some(tolerance),
        zstd_src_size: size as u64,
    };
    Ok((codec, zstd_data))
}

fn enc_zstd(data: &[f32], compression_level: u8) -> Result<(Codec, Vec<u8>)> {
    let data = bincode::encode_to_vec::<_, Configuration>(data, Configuration::default())?;
    let mut zstd = zstd::bulk::Compressor::new(compression_level as i32)?;
    zstd.multithread(256)?;
    let codec = Codec {
        zfp: None,
        zstd_src_size: data.len() as u64,
    };
    Ok((codec, zstd.compress(&data)?))
}

fn enc_zfp(data: &[f32], tolerance: f64) -> Result<Vec<u8>> {
    let arr = ArrayView1::from(data);
    let mode = if tolerance == 0.0 {
        numcodecs_zfp::ZfpCompressionMode::Reversible
    } else {
        numcodecs_zfp::ZfpCompressionMode::FixedAccuracy {
            tolerance: tolerance,
        }
    };
    Ok(numcodecs_zfp::compress(arr, &mode)?)
}