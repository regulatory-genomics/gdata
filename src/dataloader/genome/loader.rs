use anyhow::{ensure, Context, Result};
use bed_utils::bed::GenomicRange;
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{Array2, Array3, Axis};
use numpy::{PyArray2, PyArray3};
use pyo3::{prelude::*, py_run};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::path::PathBuf;
use std::str::FromStr;

use crate::dataloader::genome::data_store::{decode_nucleotide, DataStore, DataStoreReadOptions};
use super::super::generic::PrefethIterator;

/** A dataloader for genomic data, allowing for efficient retrieval of genomic
    sequences and their associated values.

    This object provides an iterator over genomic data chunks, enabling batch
    retrieval of genomic sequences and their associated values.
    The iterator yields tuples of (sequences, values) or (tag, sequences, values),
    depending on whether a tag is provided.
    Sequences has shape (batch_size, sequence_length), and values has shape
    (batch_size, sequence_length / resolution, num_tracks).

    Parameters
    ----------
    location : Path
        The path to the genomic data directory.
    batch_size : int
        The number of genomic sequences to retrieve in each batch (default is 8).
    resolution : Optional[int]
        The resolution of the genomic data. If not provided, it defaults to the dataset's resolution.
        If the resolution is provided, it must be a multiple of the dataset's resolution.
        The values will be aggregated (by taking the average) to this resolution.
    trim_target: Optional[int]
        Trim both ends of the target vector according to the `trim_target` parameter.
        As a result, the length of the values will be reduced by `2 * trim_target`.
        The unit of `trim_target` is base pairs, and it must be a multiple of the resolution.
        Note this only affects the values, not the sequences. The sequences will always
        have the full length as defined in the dataset.
        This is useful when you want to compute the loss on only the central part of the sequence.
        This is because the edges of the sequence may contain
        padding or other artifacts that should not be considered in the loss computation.
    scale : Optional[float]
        Scale the values by this factor. If not provided, no scaling is applied.
    clamp_max : Optional[float]
        Clamp the values to this maximum value. If not provided, no clamping is applied.
        If `scale` is also provided, the clamping will be applied after scaling.
    window_size : Optional[int]
        The window size for retrieving genomic sequences. The loader's window size
        can be different from the underlying dataset's window size so that the same
        dataset can be used with different window sizes. However, there are two
        restrictions: (1) The dataset's window size must be a multiple of the loader's window size;
        (2) The loader's window size must be a multiple of the dataset's resolution.
    shuffle : bool
        If True, the data will be shuffled before being returned. Default is False.
    seq_as_string : bool
        If True, sequences will be returned as strings instead of numpy integer arrays.
        This is useful for cases where you want to work with the sequences as text,
        such as for visualization or text-based analysis.
    n_jobs: int
        The number of chunks to prefetch for efficient data loading.
        The chunks here refer to the data chunks stored in the GenomeDataBuilder.
        This allows for asynchronous loading of data, improving performance during training or inference.
        But it will increase memory usage, the memory usage will be approximately
        `2 * prefetch * memory_of_chunk`.
    random_seed : int
        The random seed for shuffling the data. Default is 2025.

    See Also
    --------
    GenomeDataBuilder
    GenomeDataLoaderMap

    Examples
    --------
    >>> from gdata import as GenomeDataLoader
    >>> loader = GenomeDataLoader("test_genome", trim_target=40_960)
    >>> region = 'chr11:35041782-35238390'
    >>> tracks = ['DNase:CD14-positive monocyte', 'DNase:keratinocyte', 'ChIP-H3K27ac:keratinocyte']
    >>> loader.plot(region, tracks, savefig="signal.png")

    .. image:: /_static/images/genome_signal.png
        :align: center
*/
#[pyclass]
#[derive(Debug, Clone)]
pub struct GenomeDataLoader {
    data_store: DataStore,
    segments: Vec<GenomicRange>,
    #[pyo3(get, set)]
    batch_size: usize,
    shuffle: bool,
    seq_as_string: bool,
    n_jobs: usize,
    rng: ChaCha12Rng,
}

impl std::fmt::Display for GenomeDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(
            f,
            "GenomeDataLoader ({} segments x {} tracks):",
            self.segments.len(),
            self.tracks().len()
        )?;
        write!(
            f,
            "    window_size = {}, resolution = {}, batch_size = {}, target_length = {}",
            self.window_size(),
            self.resolution(),
            self.batch_size,
            self.data_store
                .read_opts
                .value_length
                .unwrap_or(self.data_store.sequence_length)
        )?;
        Ok(())
    }
}

impl GenomeDataLoader {
    pub fn len(&self) -> usize {
        let mut n = self.segments.len();
        if let Some(split_size) = self.data_store.read_opts.split_size {
            n *= (self.data_store.sequence_length / split_size) as usize
        }
        n / self.batch_size + if n % self.batch_size > 0 { 1 } else { 0 }
    }

    pub fn set_target_length(&mut self, target_length: u32) -> Result<()> {
        self.data_store.set_value_length(target_length)
    }

    pub fn set_window_size(&mut self, window_size: u32) -> Result<()> {
        self.data_store.set_split_size(window_size)
    }

    pub fn intersection(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        todo!()
    }

    pub fn difference(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        todo!()
    }

    pub fn iter(&mut self) -> GenomeDataLoaderIter {
        let iter = PrefethIterator::new(
            self.data_store.par_iter(self.batch_size, self.n_jobs),
            self.n_jobs * 2,
        );

        GenomeDataLoaderIter {
            iter,
            seq_as_string: self.seq_as_string,
        }
    }
}

#[pymethods]
impl GenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (location, *,
            batch_size=8, resolution=None, target_length=None, scale=None, clamp_max=None,
            window_size=None, shuffle=false, seq_as_string=false, n_jobs=8,
            random_seed=2025,
        ),
        text_signature = "($self, location, *,
            batch_size=8, resolution=None, target_length=None, scale=None, clamp_max=None,
            window_size=None, shuffle=False, seq_as_string=False, n_jobs=8,
            random_seed=2025)"
    )]
    pub fn new(
        location: PathBuf,
        batch_size: usize,
        resolution: Option<u32>,
        target_length: Option<u32>,
        scale: Option<f32>,
        clamp_max: Option<f32>,
        window_size: Option<u32>,
        shuffle: bool,
        seq_as_string: bool,
        n_jobs: usize,
        random_seed: u64,
    ) -> Result<Self> {
        let store_opts = DataStoreReadOptions {
            shift: 0,
            value_length: target_length,
            split_size: window_size,
            read_resolution: resolution,
            scale_value: scale.map(|x| bf16::from_f32(x)),
            clamp_value_max: clamp_max.map(|x| bf16::from_f32(x)),
        };

        let data_store = DataStore::open(location, store_opts)?;
        let segments = data_store.index.keys().cloned().collect();

        let loader = Self {
            data_store,
            segments,
            batch_size,
            shuffle,
            seq_as_string,
            n_jobs,
            rng: ChaCha12Rng::seed_from_u64(random_seed),
        };

        Ok(loader)
    }

    /** Returns the track names in the dataset.

       This method retrieves all keys from the dataset, which are typically the names of files
       containing genomic data. The keys are sorted alphabetically.

       Returns
       -------
       list[str]
           A sorted list of keys as strings.
    */
    #[getter]
    pub fn tracks(&self) -> Vec<String> {
        self.data_store
            .data_keys
            .iter()
            .cloned()
            .collect::<Vec<_>>()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    #[getter]
    pub fn segments(&self) -> Vec<String> {
        self.segments.iter().map(|x| x.pretty_show()).collect()
    }

    /** The length of input sequences in base pairs.

       This method returns the length of the input sequences in base pairs.
       It is determined by the `split_size` option in the read options of the data store.
       If `split_size` is not set, it defaults to the sequence length of the data store.

       Returns
       -------
       int
           The length of input sequences in base pairs.
    */
    #[getter]
    fn window_size(&self) -> u32 {
        self.data_store
            .read_opts
            .split_size
            .unwrap_or(self.data_store.sequence_length)
    }

    /** The length of the target vector in base pairs.

       Returns
       -------
       int
           The length of the target vector in base pairs.
    */
    #[getter]
    fn target_length(&self) -> u32 {
        self.data_store
            .read_opts
            .value_length
            .unwrap_or(self.window_size())
    }

    #[getter]
    fn resolution(&self) -> u32 {
        self.data_store.out_resolution
    }

    /** Returns the data indexer for accessing genomic data.

        This method provides an indexer that allows access to the genomic data
        associated with the sequences. It can be used to retrieve values for specific
        genomic regions and tracks.

        Returns
        -------
        _DataIndexer
            An indexer for accessing genomic data.
    */
    #[getter]
    fn data(slf: PyRef<'_, Self>) -> DataIndexer {
        DataIndexer(slf.into())
    }

    /** Creating a new genomic data loader in which the regions intersect with the specified ones.

        This method allows you to subset the genomic data by intersecting it with
        specified regions.
        The dataset will be updated to only include the regions that intersect with the provided ones.

        Parameters
        ----------
        regions : list[str]
            A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

        See Also
        --------
        difference : For creating a loader with differing regions.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` that contains only the data for the specified regions.
    */
    #[pyo3(
        name = "intersection",
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn intersection_py(&self, regions: Vec<String>) -> Self {
        self.intersection(
            regions.into_iter().map(|r| {
                GenomicRange::from_str(&r).expect(&format!("Invalid genomic range: {}", r))
            }),
        )
    }

    /** Creating a new genomic data loader in which the regions differ from the specified ones.

        This method allows you to subset the genomic data by removing the specified regions.

        Parameters
        ----------
        regions : list[str]
            A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

        See Also
        --------
        intersection : For creating a loader with intersecting regions.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` that contains only the data for the regions
            that do not intersect with the specified ones.
    */
    #[pyo3(
        name = "difference",
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn difference_py(&self, regions: Vec<String>) -> Self {
        self.difference(
            regions.into_iter().map(|r| {
                GenomicRange::from_str(&r).expect(&format!("Invalid genomic range: {}", r))
            }),
        )
    }

    /** Create a copy of the GenomeDataLoader.

        This method creates a new instance of `GenomeDataLoader` with the same configuration
        as the current instance. It is useful for creating independent copies of the loader
        that can be modified without affecting the original.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` with the same configuration.
    */
    fn copy(&self) -> Self {
        self.clone()
    }

    /** Plots the genomic signal for a specified region and tracks.

        This method generates a plot of the genomic signal for the specified region and tracks.
        If `savefig` is provided, it saves the plot to the specified file; otherwise, it displays the plot.

        Parameters
        ----------
        region : str
            The genomic region to plot, in the format 'chr:start-end'.
        tracks : list[str]
            A list of track names to plot.
        savefig : Optional[PathBuf]
            If provided, saves the plot to this file instead of displaying it.

        Returns
        -------
        None
    */
    #[pyo3(
        signature = (
            region, tracks, *, savefig=None,
        ),
        text_signature = "($self, region, tracks, *, savefig=None)"
    )]
    fn plot(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        region: &str,
        tracks: Bound<'_, PyAny>,
        savefig: Option<PathBuf>,
    ) -> Result<()> {
        let trim = (slf.window_size() - slf.target_length()) / 2;
        let data_indexer = Self::data(slf);
        let key = (region, &tracks).into_pyobject(py)?.into_any();
        let signal_values = data_indexer.__getitem__(py, key)?;
        let track_names = extract_string_list(tracks)?;

        let py_code = r#"
from matplotlib import pyplot as plt
height_per_track = 1.5
width = 8

start, end = region.split(":")[1].split('-')
start = int(start) + trim
end = int(end) - trim
signal_values = signal_values.T
n_tracks, n_points = signal_values.shape
fig, axes = plt.subplots(n_tracks, 1, figsize=[width, n_tracks * height_per_track], sharex=True)

if n_tracks == 1:
    axes = [axes]

for i, (ax, signal) in enumerate(zip(axes, signal_values)):
    ax.fill_between(range(n_points), 0, signal, color='black')
    ax.set_title(track_names[i], fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)

axes[-1].set_xticks([0, n_points - 1])
axes[-1].set_xticklabels([str(start), str(end)])
axes[-1].set_xlabel(region)

plt.tight_layout()
if savefig is None:
    plt.show()
else:
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
"#;
        py_run!(py, trim region track_names signal_values savefig, py_code);

        Ok(())
    }

    #[pyo3(
        signature = (*, bins=10000, limit=100, min=None, log=false, savefig=None),
        text_signature = "($self, *, bins=10000, limit=100, min=None, log=False, savefig=None)"
    )]
    fn hist(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        bins: usize,
        limit: usize,
        min: Option<f32>,
        log: bool,
        savefig: Option<PathBuf>,
    ) -> Vec<((f32, f32), usize)> {
        let (lo, hi) = slf
            .iter()
            .take(limit)
            .flat_map(|(_, values)| {
                values.into_iter().flat_map(|mut x| {
                    if min.is_some() && x < min.unwrap() {
                        None
                    } else {
                        if log {
                            x = (x + 1.0).ln();
                        }
                        Some(x)
                    }
                })
            })
            .minmax()
            .into_option()
            .unwrap();

        let bin_size = (hi - lo) / bins as f32;
        let mut histogram = vec![0; bins];
        slf.iter().take(limit).for_each(|(_, values)| {
            for mut value in values.into_iter() {
                if min.is_some() && value < min.unwrap() {
                    continue;
                }
                if log {
                    value = (value + 1.0).ln();
                }
                let bin_index = ((value - lo) / bin_size).floor() as usize;
                if bin_index < bins {
                    histogram[bin_index] += 1;
                }
            }
        });

        let result: Vec<_> = histogram
            .into_iter()
            .enumerate()
            .flat_map(|(i, count)| {
                if count == 0 {
                    None
                } else {
                    let bin_start = lo + i as f32 * bin_size;
                    let bin_end = bin_start + bin_size;
                    Some(((bin_start, bin_end), count))
                }
            })
            .collect();

        let (xs, ys): (Vec<_>, Vec<_>) = result
            .iter()
            .map(|((x1, x2), y)| ((x1 + x2) / 2.0, *y))
            .unzip();

        let py_code = r#"
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, ys, 'b-')
if log:
    ax.set_xlabel("log(value + 1)")
else:
    ax.set_xlabel("value")
plt.tight_layout()
if savefig is None:
    plt.show()
else:
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
"#;
        py_run!(py, xs ys log savefig, py_code);
        result
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> GenomeDataLoaderIter {
        slf.iter()
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pyclass]
pub struct GenomeDataLoaderIter {
    iter: PrefethIterator<(Array2<u8>, Array3<f32>)>,
    seq_as_string: bool,
}

impl Iterator for GenomeDataLoaderIter {
    type Item = (Array2<u8>, Array3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let (mut seq, val) = self.iter.next()?;
        if self.seq_as_string {
            seq.mapv_inplace(|x| decode_nucleotide(x).unwrap());
        }
        Some((seq, val))
    }
}

#[pymethods]
impl GenomeDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;
        let values = PyArray3::from_owned_array(py, values);
        let result = if slf.seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

#[pyclass]
struct DataIndexer(Py<GenomeDataLoader>);

#[pymethods]
impl DataIndexer {
    fn __getitem__<'a>(
        &'a self,
        py: Python<'a>,
        key: Bound<'a, PyAny>,
    ) -> Result<(Vec<String>, Bound<'a, PyArray3<f32>>)> {
        let loader = self.0.borrow(py);
        let (seq_index, mut data_index): (Bound<'_, PyAny>, Bound<'_, PyAny>) = key.extract()?;
        if !data_index.is_instance_of::<pyo3::types::PyList>() {
            data_index = vec![data_index].into_pyobject(py)?.into_any();
        }

        let (seq, val) = if let Ok(region) = seq_index.extract::<String>() {
            let region = GenomicRange::from_str(&region).unwrap();
            loader.data_store.read(&region).unwrap()
        } else {
            let idx: usize = seq_index.extract()?;
            loader.data_store.read_at(idx).unwrap()
        };

        let idx: Vec<usize> = if let Ok(j_) = data_index.extract::<Vec<String>>() {
            j_.into_iter()
                .map(|x| loader.data_store.data_keys.get_index_of(&x).unwrap())
                .collect()
        } else {
            data_index.extract()?
        };

        let val: Array3<bf16> = val.into();
        let val = val.select(Axis(2), &idx).mapv(|x| x.to_f32());

        Ok((seq_to_string(&seq.into()), PyArray3::from_owned_array(py, val)))
    }
}

/** A dictionary-like class for loading multiple genomic datasets simultaneously.

    This class allows you to load and iterate over multiple genomic datasets simultaneously,
    each identified by a unique tag. It ensures that all datasets share the same genomic segments,
    which is a requirement for consistent data processing.

    Note
    ----
    The tracks in different loaders may have different resolutions, but the segments must be the same.
    `GenomeDataLoaderMap` ensures that records correspond to the same genomic segments across all datasets,
    but there is no guarantee that the data values returned by different loaders will be aligned, e.,.,
    they may have different `resolution` or `trim_target`.

    Parameters
    ----------
    loaders: dict[str, GenomeDataLoader]
        A dictionary mapping tags to `GenomeDataLoader` instances.
    batch_size : Optional[int]
        Optional parameter to specify the batch size for loading genomic sequences.
        If not provided, it defaults to the minimum batch size across all loaders.
    target_length: Optional[int]
        Optional parameter to specify the target length for all loaders.
        If not provided, each loader will use its own target length.
    window_size : Optional[int]
        Optional parameter to specify the window size for all loaders.
    seq_as_string : bool
        If True, sequences will be returned as strings instead of numpy integer arrays.
        This is useful for cases where you want to work with the sequences as text,
        such as for visualization or text-based analysis.

    See Also
    --------
    GenomeDataLoader
    GenomeDataBuilder
*/
#[pyclass]
#[derive(Debug, Clone)]
pub struct GenomeDataLoaderMap(IndexMap<String, GenomeDataLoader>);

impl std::fmt::Display for GenomeDataLoaderMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(
            f,
            "MultiGenomeDataLoader with keys: {}",
            self.0.keys().map(|x| "'".to_owned() + x + "'").join(", "),
        )?;
        Ok(())
    }
}

impl GenomeDataLoaderMap {
    pub fn len(&self) -> usize {
        self.0[0].len()
    }

    pub fn iter(&mut self) -> MultiDataLoaderIter {
        let iter = self
            .0
            .iter_mut()
            .map(|(tag, loader)| (tag.clone(), loader.iter()))
            .collect();
        MultiDataLoaderIter(iter)
    }
}

#[pymethods]
impl GenomeDataLoaderMap {
    #[new]
    #[pyo3(
        signature = (
            loaders, *, batch_size=None, target_length=None, window_size=None, seq_as_string=false,
        ),
        text_signature = "($self, loaders, *, batch_size=None, target_length=None, window_size=None, seq_as_string=False)"
    )]
    pub fn new(
        mut loaders: IndexMap<String, GenomeDataLoader>,
        batch_size: Option<usize>,
        target_length: Option<u32>,
        window_size: Option<u32>,
        seq_as_string: bool,
    ) -> Result<Self> {
        ensure!(
            !loaders.is_empty(),
            "At least one GenomeDataLoader must be provided"
        );
        ensure!(
            loaders.values().map(|loader| loader.segments()).all_equal(),
            "All genome data loaders must have the same segments"
        );

        let batch_size = batch_size.unwrap_or_else(|| {
            loaders
                .values()
                .map(|loader| loader.batch_size)
                .min()
                .unwrap()
        });

        loaders.values_mut().for_each(|loader| {
            loader.seq_as_string = seq_as_string;
            loader.batch_size = batch_size;
            target_length.map(|t| {
                loader.set_target_length(t).unwrap();
            });
            window_size.map(|w| {
                loader.set_window_size(w).unwrap();
            });
        });

        ensure!(
            loaders
                .values()
                .map(|loader| loader.window_size())
                .all_equal(),
            "All genome data loaders must have the same window size",
        );

        Ok(Self(loaders.into_iter().map(|(k, v)| (k, v)).collect()))
    }

    /** Returns a dictionary mapping dataset tags to the number of tracks.

      Returns
      -------
      dict[str, int]
          A dictionary where keys are dataset tags and values are the number of tracks.
    */
    #[getter]
    fn n_tracks(&self) -> IndexMap<String, usize> {
        self.0
            .iter()
            .map(|(k, v)| {
                let n_tracks = v.tracks().len();
                (k.clone(), n_tracks)
            })
            .collect()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    #[getter]
    fn segments(&self) -> Vec<String> {
        self.0[0].segments()
    }

    /** batch size of the dataloader.
     */
    #[getter]
    fn batch_size(&self) -> usize {
        self.0[0].batch_size
    }

    /** Creates a new genomic data loader based on specified regions.

       This method allows you to subset the genomic data by intersecting it with
       specified regions across all loaders in the map.
       The dataset will be updated to only include the regions that intersect with the provided ones.

       Parameters
       ----------
       regions : list[str]
           A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

       Returns
       -------
       GenomeDataLoaderMap
           A new instance of `GenomeDataLoaderMap` that contains only the data for the specified regions.
    */
    #[pyo3(
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn intersection(&self, regions: Vec<String>) -> Result<Self> {
        let result = self
            .0
            .iter()
            .map(|(tag, loader)| {
                let new_loader = loader.intersection_py(regions.clone());
                Ok((tag.clone(), new_loader))
            })
            .collect::<Result<IndexMap<_, _>>>();
        Ok(Self(result?))
    }

    /** Creating a new genomic data loader in which the regions differ from the specified ones.

        This method allows you to subset the genomic data by removing the specified regions.

        Parameters
        ----------
        regions : list[str]
            A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

        See Also
        --------
        intersection : For creating a loader with intersecting regions.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` that contains only the data for the regions
            that do not intersect with the specified ones.
    */
    #[pyo3(
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn difference(&self, regions: Vec<String>) -> Result<Self> {
        let result = self
            .0
            .iter()
            .map(|(tag, loader)| {
                let new_loader = loader.difference_py(regions.clone());
                Ok((tag.clone(), new_loader))
            })
            .collect::<Result<IndexMap<_, _>>>();
        Ok(Self(result?))
    }

    /** Returns the keys of the dataloader.

       Returns
       -------
       list[str]
           A list of keys representing the genomic segments in the dataset.
    */
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> MultiDataLoaderIter {
        slf.iter()
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pyclass]
pub struct MultiDataLoaderIter(IndexMap<String, GenomeDataLoaderIter>);

impl Iterator for MultiDataLoaderIter {
    type Item = (Array2<u8>, IndexMap<String, Array3<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut seqs = None;
        let data: Option<_> = self
            .0
            .iter_mut()
            .map(|(tag, iter)| {
                let (s, d) = iter.next()?;
                if let Some(s_) = seqs.as_ref() {
                    assert_eq!(s_, &s, "All sequences must be the same");
                } else {
                    seqs = Some(s);
                }
                Some((tag.clone(), d))
            })
            .collect();

        Some((seqs?, data?))
    }
}

#[pymethods]
impl MultiDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;

        let values = values
            .into_iter()
            .map(|(tag, v)| (tag, PyArray3::from_owned_array(py, v)))
            .collect::<IndexMap<_, _>>();

        let result = if slf.0[0].seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

/** This class combines multiple `GenomeDataLoaderMap` instances into a single loader.

    It allows for the simultaneous loading of genomic data from multiple species.
    The resulting loader will iterate over all datasets in an alternating fashion.

    Parameters
    ----------
    loaders: list[GenomeDataLoaderMap]
        A list of `GenomeDataLoaderMap` instances to combine.
    batch_size : Optional[int]
        Optional parameter to specify the batch size for loading genomic sequences.
    shuffle : Optional[bool]
        Optional parameter to specify whether to shuffle the data across all loaders.
*/
#[pyclass]
#[derive(Debug, Clone)]
pub struct CatGenomeDataLoader(Vec<GenomeDataLoaderMap>);

impl CatGenomeDataLoader {
    pub fn len(&self) -> usize {
        self.0.iter().map(|loader| loader.len()).sum()
    }

    pub fn iter(&mut self) -> MultiGenomeIter {
        let iters = self
            .0
            .iter_mut()
            .map(|loader| loader.iter())
            .collect::<Vec<_>>();
        MultiGenomeIter { iters, pos: 0 }
    }
}

#[pymethods]
impl CatGenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (loaders, *, batch_size=None, shuffle=None),
        text_signature = "($self, loaders, *, batch_size=None, shuffle=None)"
    )]
    pub fn new(
        mut loaders: Vec<GenomeDataLoaderMap>,
        batch_size: Option<usize>,
        shuffle: Option<bool>,
    ) -> Result<Self> {
        ensure!(
            !loaders.is_empty(),
            "At least one GenomeDataLoaderMap must be provided"
        );

        if let Some(bs) = batch_size {
            loaders.iter_mut().for_each(|loader| {
                loader.0.values_mut().for_each(|l| l.batch_size = bs);
            });
        }

        if let Some(shuffle) = shuffle {
            loaders.iter_mut().for_each(|loader| {
                loader.0.values_mut().for_each(|l| l.shuffle = shuffle);
            });
        }

        Ok(Self(loaders))
    }

    /** Returns a dictionary mapping dataset tags to the number of tracks.

      Returns
      -------
      dict[str, int]
          A dictionary where keys are dataset tags and values are the number of tracks.
    */
    #[getter]
    fn n_tracks(&self) -> IndexMap<String, usize> {
        IndexMap::from_iter(
            self.0
                .iter()
                .flat_map(|loader| loader.n_tracks().into_iter()),
        )
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> MultiGenomeIter {
        slf.iter()
    }
}

#[pyclass]
pub struct MultiGenomeIter {
    iters: Vec<MultiDataLoaderIter>,
    pos: usize,
}

impl Iterator for MultiGenomeIter {
    type Item = (Array2<u8>, IndexMap<String, Array3<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iters.is_empty() {
            return None;
        }

        self.pos %= self.iters.len();
        if let Some(item) = self.iters[self.pos].next() {
            self.pos += 1;
            Some(item)
        } else {
            self.iters.remove(self.pos);
            self.next()
        }
    }
}

#[pymethods]
impl MultiGenomeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;
        let values = values
            .into_iter()
            .map(|(tag, v)| (tag, PyArray3::from_owned_array(py, v)))
            .collect::<IndexMap<_, _>>();

        let result = if slf.iters[0].0[0].seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}


fn extract_string_list(str: Bound<'_, PyAny>) -> Result<Vec<String>> {
    if let Ok(s) = str.extract::<String>() {
        Ok(vec![s])
    } else {
        str.extract::<Vec<String>>()
            .with_context(|| "Failed to extract string list from PyAny")
    }
}

pub(crate) fn seq_to_string(seq: &Array2<u8>) -> Vec<String> {
    seq.rows()
        .into_iter()
        .map(|row| String::from_utf8(row.to_vec()).unwrap())
        .collect()
}