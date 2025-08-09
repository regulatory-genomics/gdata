mod bigwig;
mod bam;

use pyo3::prelude::*;

#[pymodule]
pub(crate) fn register_utils(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let utils = PyModule::new(parent_module.py(), "utils")?;

    utils.add_function(wrap_pyfunction!(bigwig::bw_to_w5z, &utils)?)?;
    utils.add_function(wrap_pyfunction!(bam::bam_cov, &utils)?)?;

    parent_module.add_submodule(&utils)
}