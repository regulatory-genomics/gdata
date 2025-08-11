pub mod w5z;
pub mod dataloader;
pub mod utils;

use pyo3::prelude::*;
use std::io::Write;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn gdata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::builder()
        .format(|buf, record| {
            let timestamp = buf.timestamp();
            let style = buf.default_level_style(record.level());
            writeln!(
                buf,
                "[{timestamp} {style}{}{style:#}] {}",
                record.level(),
                record.args()
            )
        })
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .unwrap();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<w5z::W5Z>()?;
    m.add_class::<dataloader::DataLoader>()?;
    m.add_class::<dataloader::genome::GenomeDataBuilder>()?;
    m.add_class::<dataloader::genome::GenomeDataLoader>()?;
    m.add_class::<dataloader::genome::GenomeDataLoaderMap>()?;
    m.add_class::<dataloader::genome::CatGenomeDataLoader>()?;

    utils::register_utils(m)?;

    Ok(())
}
