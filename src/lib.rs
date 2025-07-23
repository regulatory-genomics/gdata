mod bam;
mod bigwig;
mod w5z;
mod dataloader;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _gdata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<w5z::W5Z>()?;
    m.add_class::<dataloader::GenomeDataBuilder>()?;
    m.add_class::<dataloader::GenomeDataLoader>()?;
    m.add_class::<dataloader::GenomeDataLoaderMap>()?;

    m.add_function(wrap_pyfunction!(bigwig::bw_to_w5z, m)?)?;

    m.add_function(wrap_pyfunction!(bam::bam_cov, m)?)?;
    Ok(())
}