use anyhow::{bail, Result};
use bincode::config::Configuration;
use hdf5::{File, Group};
use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::path::{Path, PathBuf};

/** W5Z object is a zfp compressed basepair resolution data format for representing genome-wide
 * coverage data, such as ChIP-seq, ATAC-seq, and other genomic signals.
*/
#[pyclass]
#[repr(transparent)]
pub struct W5Z {
    inner: File,
}

impl W5Z {
    pub fn open(filename: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { inner: File::open(filename)? })
    }
 
    pub fn get(&self, key: &str) -> Result<Array1<f32>> {
        let group = self.inner.group("/")?;
        let dataset = group.dataset(key)?;
        let size = dataset.attr("zstd_size")?.read_scalar::<u64>()?;
        let zfp = dataset.attr("zfp")?.read_scalar::<bool>()?;
        let arr = decode_z(&dataset.read_1d()?, zfp, size as usize)?;
        Ok(arr)
    }

    pub fn add(&self, key: &str, value: &[f32]) -> Result<()> {
        let group = self.inner.group("/")?;
        write_z(
            &group,
            key,
            value,
            &mut Some(false), // zfp is not set yet
            0.0, // default precision
            19, // default compression level
        )?;
        Ok(())
    }

    pub fn contains(&self, key: &str) -> Result<bool> {
        let group = self.inner.group("/")?;
        match group.dataset(key) {
            Ok(_) => Ok(true),
            _ => Ok(false),
        }
    }
}

#[pymethods]
impl W5Z {
    #[new]
    #[pyo3(
        signature = (filename, mode="r"),
        text_signature = "($self, filename, mode='r')"
    )]
    pub fn new(filename: PathBuf, mode: &str) -> Result<Self> {
        let inner = match mode {
            "r" => File::open(filename)?,
            "w" => {
                if filename.exists() {
                    bail!("File already exists: {}", filename.display());
                }
                File::create(filename)?
            },
            _ => bail!("Invalid mode: {}", mode),
        };
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
        let arr = self.get(key)?;
        Ok(PyArray1::from_owned_array(py, arr))
    }

    fn __setitem__(&self, key: &str, value: PyReadonlyArray1<'_, f32>) -> Result<()> {
        self.add(key, value.as_slice()?)
    }

    fn list_attrs(&self) -> Result<Vec<String>> {
        let group = self.inner.group("/")?;
        let attrs = group.attr_names()?;
        Ok(attrs)
    }

    fn attr(&self, key: &str) -> Result<f64> {
        let group = self.inner.group("/")?;
        let attr = group.attr(key)?;
        Ok(attr.read_scalar::<f64>()?)
    }

    fn verify(&self, py: Python) -> Result<f64> {
        let real_s = self.inner.attr("sum")?.read_scalar::<f64>()?;
        let stat = self.compute_stat(py)?;
        Ok((stat.sum() - real_s).abs() / real_s)
    }

    fn close(&self) -> Result<()> {
        self.inner.clone().close()?;
        Ok(())
    }

    fn update_stat(&self, py: Python) -> Result<()> {
        let group = self.inner.group("/")?;
        let stats = self.compute_stat(py)?;
        stats.write_metadata(&group)?;
        Ok(())
    }

    fn compute_stat(&self, py: Python) -> Result<Statistics> {
        let mut stats = Statistics::new();
        for key in self.keys()? {
            let data = self.__getitem__(py, &key)?;
            for x in data.try_iter()? {
                stats.add(x?.extract::<f32>()?);
            }
        }
        Ok(stats)
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __str__(&self) -> String {
        format!("W5Z object with keys: {}", self.keys().unwrap().join(", "))
    }
}

#[pyclass]
pub struct Statistics {
    min: f64,
    max: f64,
    sum_x: f64,
    sum_x2: f64,
    n: usize,
}

impl Statistics {
    pub fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum_x: 0.0,
            sum_x2: 0.0,
            n: 0,
        }
    }

    pub fn add(&mut self, x: f32) {
        if x.is_finite() {
            self.min = self.min.min(x.into());
            self.max = self.max.max(x.into());
            self.sum_x += x as f64;
            self.sum_x2 += (x * x) as f64;
            self.n += 1;
        }
    }

    pub fn sum(&self) -> f64 {
        self.sum_x
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn mean(&self) -> f64 {
        self.sum_x / self.n as f64
    }

    pub fn stddev(&self) -> f64 {
        let mean = self.mean();
        (self.sum_x2 / self.n as f64) - (mean * mean)
    }

    pub fn write_metadata(&self, group: &Group) -> Result<()> {
        let attr = group.new_attr::<f64>().create("mean")?;
        attr.write_scalar(&self.mean())?;
        let attr = group.new_attr::<f64>().create("min")?;
        attr.write_scalar(&self.min())?;
        let attr = group.new_attr::<f64>().create("max")?;
        attr.write_scalar(&self.max())?;
        let attr = group.new_attr::<f64>().create("stddev")?;
        attr.write_scalar(&self.stddev())?;
        let attr = group.new_attr::<f64>().create("sum")?;
        attr.write_scalar(&self.sum())?;
        Ok(())
    }
}



pub struct Codec {
    pub zfp: Option<f64>,
    pub zstd_src_size: u64,
}

pub fn write_z(h5: &Group, name: &str, data: &[f32], zfp: &mut Option<bool>, precision: f64, compression_level: u8) -> Result<usize> {
    let (codec, data) = encode_z(data, zfp.clone(), precision, compression_level).unwrap();
    h5.new_dataset::<u8>()
        .shape([data.len()])
        .create(name)
        .unwrap()
        .write(&data)
        .unwrap();
    let dataset = h5.dataset(name)?;
    dataset
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("dtype")?
        .write_scalar(&"float32".parse::<hdf5::types::VarLenUnicode>().unwrap())?;
    dataset
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("encoding")?
        .write_scalar(&"zfp".parse::<hdf5::types::VarLenUnicode>().unwrap())?;
    dataset
        .new_attr::<u64>()
        .create("zstd_size")?
        .write_scalar(&codec.zstd_src_size)?;

    let use_zfp = codec.zfp.is_some();
    dataset
        .new_attr::<bool>()
        .create("zfp")?
        .write_scalar(&use_zfp)?;
    if use_zfp {
        dataset
            .new_attr::<f64>()
            .create("tolerance")?
            .write_scalar(&codec.zfp.unwrap())?;
    }

    // Update zfp 
    if zfp.is_none() {
        log::info!("Using ZFP compression: {}", use_zfp);
        *zfp = Some(use_zfp);
    }
        
    Ok(data.len())
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