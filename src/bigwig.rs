use anyhow::Result;
use bigtools::BigWigRead;
use hdf5::{File, Group};
use indicatif::ProgressStyle;
use itertools::Itertools;
use ndarray::{Array1, ArrayView1};
use pyo3::prelude::*;
use reqwest::Url;
use std::cmp::min;
use std::{
    io::{Read, Seek, Write},
    path::PathBuf,
};
use tokio_stream::StreamExt;

use indicatif::ProgressBar;
use reqwest::Client;

pub struct Stats {
    min: f64,
    max: f64,
    sum_x: f64,
    sum_x2: f64,
    n: usize,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
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

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(
    signature = (input, output, *, compression_level=9, precision=0.0, verify=false),
    text_signature = "(input, output, *, compression_level=9, precision=0.0, verify=False)",
)]
pub fn bw_to_w5z(
    input: &str,
    output: PathBuf,
    compression_level: u8,
    precision: f64,
    verify: bool,
) -> Result<()> {
    hdf5::filters::blosc_set_nthreads(128);
    let h5 = File::create(output)?;

    if is_url(input) {
        tokio::runtime::Runtime::new()?.block_on(async {
            let mut temp = tempfile::spooled_tempfile(4 * 1024 * 1024 * 1024);
            download_file(input, &mut temp).await.unwrap();
            temp.rewind()?;
            let mut bw = BigWigRead::open(temp)?;
            save_bw(&mut bw, &h5, compression_level, precision, verify)
        })?;
    } else {
        let file = std::fs::File::open(input)?;
        let mut bw = BigWigRead::open(file)?;
        save_bw(&mut bw, &h5, compression_level, precision, verify)?;
    }

    Ok(())
}

fn save_bw<R: Read + Seek>(
    bw: &mut BigWigRead<R>,
    h5: &Group,
    compression_level: u8,
    precision: f64,
    verify: bool,
) -> Result<()> {
    let style = ProgressStyle::with_template(
        "[{elapsed}] {wide_bar:.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
    )
    .unwrap();
    let chromosomes: Vec<_> = bw
        .chroms()
        .iter()
        .sorted_by(|a, b| b.length.cmp(&a.length))
        .cloned()
        .collect();
    let pb = indicatif::ProgressBar::new(chromosomes.iter().map(|x| x.length as u64).sum::<u64>())
        .with_style(style);
    let mut stats = Stats::new();
    chromosomes.into_iter().for_each(|chrom| {
        let mut vals = bw.values(&chrom.name, 0, chrom.length).unwrap();
        fix_nan(&mut vals, Some(0.0));
        for &x in vals.iter() {
            stats.add(x);
        }

        write_data(
            &h5,
            &chrom.name,
            encode_zfp(&vals, precision).unwrap(),
            compression_level,
        )
        .unwrap();

        if verify {
            let data = read_data(&h5, &chrom.name).unwrap();
            assert_eq!(data.len(), vals.len());
            for (a, b) in data.iter().zip(vals.iter()) {
                assert!(((a - *b).abs() as f64) < precision);
            }
        }

        pb.inc(chrom.length as u64);
    });

    stats.write_metadata(h5)?;

    log::info!(
        "Sum: {}, Min: {}, Max: {}, Mean: {}, StdDev: {}",
        stats.sum(),
        stats.min(),
        stats.max(),
        stats.mean(),
        stats.stddev()
    );
    Ok(())
}

/// Fix NaN values in the data by replacing them with the minimum finite value.
fn fix_nan(data: &mut Vec<f32>, min: Option<f32>) {
    let d_min = match min {
        Some(min) => min,
        None => data
            .iter()
            .filter(|x| x.is_finite())
            .min_by(|a, b| {
                a.partial_cmp(b)
                    .expect(&format!("Cannot compare {} and {}", a, b))
            })
            .unwrap()
            .clone(),
    };
    for x in data.iter_mut() {
        if !x.is_finite() {
            *x = d_min;
        }
    }
}

fn read_data(h5: &Group, name: &str) -> Result<Vec<f32>> {
    let dataset = h5.dataset(name)?;
    let data: Array1<u8> = dataset.read()?;
    let decoded: Array1<f32> = numcodecs_zfp::decompress(data.as_slice().unwrap())?
        .as_typed::<f32>()
        .unwrap()
        .to_owned()
        .into_dimensionality()?;
    Ok(decoded.to_vec())
}

fn write_data(h5: &Group, name: &str, data: Vec<u8>, compression_level: u8) -> Result<()> {
    let mut builder = h5.new_dataset::<u8>();
    if compression_level > 0 {
        builder = builder.blosc_zstd(compression_level, hdf5::filters::BloscShuffle::Byte);
    }
    builder
        .shape([data.len()])
        .create(name)
        .unwrap()
        .write(&data)
        .unwrap();
    let dataset = h5.dataset(name)?;
    let attr = dataset
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("dtype")?;
    let value_: hdf5::types::VarLenUnicode = "float32".parse().unwrap();
    attr.write_scalar(&value_)?;
    let attr = dataset
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("encoding")?;
    let value_: hdf5::types::VarLenUnicode = "zfp".parse().unwrap();
    attr.write_scalar(&value_)?;
    Ok(())
}

/// Encode the data using ZFP compression with a specified precision.
fn encode_zfp(data: &[f32], precision: f64) -> Result<Vec<u8>> {
    let arr = ArrayView1::from(data);
    let mode = if precision == 0.0 {
        numcodecs_zfp::ZfpCompressionMode::Reversible
    } else {
        numcodecs_zfp::ZfpCompressionMode::FixedAccuracy {
            tolerance: precision,
        }
    };
    let encoded_data = numcodecs_zfp::compress(arr, &mode)?;
    Ok(encoded_data)
}

fn is_url(src: &str) -> bool {
    if let Ok(url) = Url::parse(src) {
        if !url.cannot_be_a_base() {
            return true;
        }
    }
    return false;
}

async fn download_file<R: Write>(url: &str, out: &mut R) -> Result<(), String> {
    // Reqwest setup
    let res = Client::new()
        .get(url)
        .send()
        .await
        .or(Err(format!("Failed to GET from '{}'", &url)))?;
    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from '{}'", &url))?;

    // Indicatif setup
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap());

    // download chunks
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.or(Err(format!("Error while downloading file")))?;
        out.write_all(&chunk)
            .or(Err(format!("Error while writing to file")))?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    return Ok(());
}
