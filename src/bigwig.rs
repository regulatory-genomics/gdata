use anyhow::Result;
use bigtools::BigWigRead;
use hdf5::{File, Group};
use indicatif::ProgressStyle;
use itertools::Itertools;
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

use crate::w5z::Statistics;

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(
    signature = (input, output, *, zfp=None, compression_level=19, tolerance=0.0),
    text_signature = "(input, output, *, zfp=None, compression_level=19, tolerance=0.0)",
)]
pub fn bw_to_w5z(
    input: &str,
    output: PathBuf,
    zfp: Option<bool>,
    compression_level: u8,
    tolerance: f64,
) -> Result<()> {
    let h5 = File::create(output)?;

    if is_url(input) {
        tokio::runtime::Runtime::new()?.block_on(async {
            let mut temp = tempfile::spooled_tempfile(4 * 1024 * 1024 * 1024);
            download_file(input, &mut temp).await.unwrap();
            temp.rewind()?;
            let mut bw = BigWigRead::open(temp)?;
            save_bw(&mut bw, &h5, zfp, compression_level, tolerance)
        })?;
    } else {
        let file = std::fs::File::open(input)?;
        let mut bw = BigWigRead::open(file)?;
        save_bw(&mut bw, &h5, zfp, compression_level, tolerance)?;
    }

    Ok(())
}

fn save_bw<R: Read + Seek>(
    bw: &mut BigWigRead<R>,
    h5: &Group,
    mut zfp: Option<bool>,
    compression_level: u8,
    precision: f64,
) -> Result<()> {
    let style = ProgressStyle::with_template(
        "[{elapsed}] {wide_bar:.cyan/blue} {percent}/100% (eta: {eta})",
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
    let mut stats = Statistics::new();
    let mut total_size = 0u64;
    let mut compressed_size = 0u64;
    chromosomes.into_iter().for_each(|chrom| {
        let mut vals = bw.values(&chrom.name, 0, chrom.length).unwrap();
        fix_nan(&mut vals, Some(0.0));
        for &x in vals.iter() {
            stats.add(x);
        }
        total_size += vals.len() as u64 * 4; // 4 bytes per f32
        compressed_size += crate::w5z::write_z(&h5, &chrom.name, &vals, &mut zfp, precision, compression_level).unwrap() as u64;

        pb.inc(chrom.length as u64);
    });

    stats.write_metadata(h5)?;

    log::info!("Compression ratio: {:.2}%", compressed_size as f64 / total_size as f64 * 100.0);
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
