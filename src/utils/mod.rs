mod bigwig;
mod bam;
mod gff;

use pyo3::prelude::*;

#[pymodule]
pub(crate) fn register_utils(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let utils = PyModule::new(parent_module.py(), "utils")?;

    utils.add_function(wrap_pyfunction!(bigwig::bw_to_w5z, &utils)?)?;
    utils.add_function(wrap_pyfunction!(bam::bam_cov, &utils)?)?;
    utils.add_function(wrap_pyfunction!(gff::read_transcripts, &utils)?)?;

    parent_module.add_submodule(&utils)
}