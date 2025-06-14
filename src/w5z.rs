use anyhow::Result;
use hdf5::File;
use ndarray::Array1;
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
        let arr = decode_zfp(&dataset.read_1d()?)?;
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

fn decode_zfp(data: &Array1<u8>) -> Result<Array1<f32>> {
    let decoded: Array1<f32> = numcodecs_zfp::decompress(data.as_slice().unwrap())?
        .as_typed::<f32>()
        .unwrap()
        .to_owned()
        .into_dimensionality()?;
    Ok(decoded)
}
