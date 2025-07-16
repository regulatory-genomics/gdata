mod bam;
mod bigwig;
mod w5z;
mod data;
mod utils;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _gdata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<w5z::W5Z>()?;
    m.add_class::<data::GenomeDataBuilder>()?;
    m.add_class::<data::GenomeDataLoader>()?;

    m.add_function(wrap_pyfunction!(bigwig::bw_to_w5z, m)?)?;

    m.add_function(wrap_pyfunction!(bam::bam_cov, m)?)?;
    Ok(())
}