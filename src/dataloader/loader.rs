use anyhow::{ensure, Context, Result};
use bed_utils::bed::GenomicRange;
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{Array, Array1, ArrayD, Dimension, Ix2};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::{prelude::*, py_run};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

use crate::dataloader::builder::GenomeDataBuilder;
use crate::dataloader::chunk::{DataChunk, Sequences};

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
    prefetch : int
        The number of batches to prefetch for efficient data loading.
        This allows for asynchronous loading of data, improving performance during training or inference.
        But it will increase memory usage, so it should be set according to the available resources.
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
    builder: GenomeDataBuilder,
    window_size: Option<u64>,
    batch_size: usize,
    trim_target: Option<usize>,
    scale: Option<f32>,
    clamp_max: Option<f32>,
    shuffle: bool,
    seq_as_string: bool,
    prefetch: usize,
    rng: ChaCha12Rng,
}

impl std::fmt::Display for GenomeDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(
            f,
            "GenomeDataLoader at '{}' with {} segments x {} tracks:",
            self.builder.location.display(),
            self.builder.seq_index.len(),
            self.builder.tracks().unwrap().len()
        )?;
        write!(
            f,
            "    window_size = {}, resolution = {}, batch_size = {}, trim_target = {}",
            self.window_size(),
            self.builder.resolution,
            self.batch_size,
            self.trim_target.unwrap_or(0),
        )?;
        Ok(())
    }
}

impl GenomeDataLoader {
    pub fn len(&self) -> usize {
        let mut n = self.builder.seq_index.len();
        let multiply_by = self.builder.window_size / self.window_size.unwrap_or(self.builder.window_size);
        n *= multiply_by as usize;
        n / self.batch_size + if n % self.batch_size > 0 { 1 } else { 0 }
    }

    pub fn set_trim_target(&mut self, trim_target: usize) {
        if trim_target >= (self.window_size() / 2) as usize {
            panic!("Trim target must be less than half of the window size");
        } else if trim_target % self.builder.resolution as usize != 0 {
            panic!(
                "Trim target must be a multiple of resolution ({})",
                self.builder.resolution
            );
        }
        self.trim_target = Some(trim_target);
    }

    pub fn intersection(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let mut new_loader = self.clone();
        let builder = &mut new_loader.builder;
        builder.seq_index = builder.seq_index.intersection(regions);
        let chromosomes = builder.seq_index.chromosomes().collect::<HashSet<_>>();
        builder
            .chrom_sizes
            .retain(|chrom, _| chromosomes.contains(chrom));
        new_loader
    }

    pub fn difference(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let mut new_loader = self.clone();
        let builder = &mut new_loader.builder;
        builder.seq_index = builder.seq_index.difference(regions);
        let chromosomes = builder.seq_index.chromosomes().collect::<HashSet<_>>();
        builder
            .chrom_sizes
            .retain(|chrom, _| chromosomes.contains(chrom));
        new_loader
    }

    pub fn iter(&mut self) -> DataLoaderIter {
        let shuffle = if self.shuffle {
            Some(&mut self.rng)
        } else {
            None
        };
        let split_data = if let Some(s) = self.window_size {
            let builder_window_size = self.builder.window_size;
            let resolution = self.builder.resolution;
            assert!(
                s < builder_window_size,
                "Loader's window size must be less than the dataset's window size ({})",
                builder_window_size,
            );
            assert!(
                s % resolution == 0,
                "Loader's window size must be a multiple of the dataset's resolution ({})",
                resolution,
            );

            if s == builder_window_size {
                None
            } else {
                Some((s / resolution) as usize)
            }
        } else {
            None
        };
        DataLoaderIter {
            iter: PrefethIterator::new(
                _DataLoaderIter {
                    batch_size: self.batch_size,
                    buffer_seq: Buffer::new(),
                    buffer_data: Buffer::new(),
                    scale: self.scale.map(bf16::from_f32),
                    clamp_max: self.clamp_max.map(bf16::from_f32),
                    chunks: self.builder.seq_index.iter_chunks(
                        split_data,
                        self.trim_target
                            .map(|t| t / self.builder.resolution as usize),
                        false,
                        shuffle,
                    ),
                },
                self.prefetch,
            ),
            seq_as_string: self.seq_as_string,
        }
    }
}

#[pymethods]
impl GenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (location, *,
            batch_size=8, trim_target=None, scale=None, clamp_max=None,
            window_size=None, shuffle=false, seq_as_string=false, prefetch=16,
            random_seed=2025,
        ),
        text_signature = "($self, location, *,
            batch_size=8, trim_target=None, scale=None, clamp_max=None,
            window_size=None, shuffle=False, seq_as_string=False, prefetch=16,
            random_seed=2025)"
    )]
    pub fn new(
        location: PathBuf,
        batch_size: usize,
        trim_target: Option<usize>,
        scale: Option<f32>,
        clamp_max: Option<f32>,
        window_size: Option<u64>,
        shuffle: bool,
        seq_as_string: bool,
        prefetch: usize,
        random_seed: u64,
    ) -> Result<Self> {
        let builder = GenomeDataBuilder::open(location)?;
        if let Some(window_size) = window_size {
            ensure!(
                builder.window_size % window_size == 0,
                "dataset's window size ({}) must be a multiple of the loader's window size",
                builder.window_size,
            );
            ensure!(
                window_size % builder.resolution == 0,
                "Loader's window size must be a multiple of the dataset's resolution ({})",
                builder.resolution
            );
        }

        let mut loader = Self {
            builder,
            window_size,
            batch_size,
            trim_target: None,
            scale,
            clamp_max,
            shuffle,
            seq_as_string,
            prefetch,
            rng: ChaCha12Rng::seed_from_u64(random_seed),
        };

        if let Some(t) = trim_target {
            loader.set_trim_target(t);
        }

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
    fn tracks(&self) -> Result<Vec<String>> {
        self.builder.tracks()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    #[getter]
    fn segments(&self) -> Vec<String> {
        self.builder
            .seq_index
            .keys()
            .map(|x| x.pretty_show())
            .collect()
    }

    #[getter]
    fn window_size(&self) -> u64 {
        self.window_size.unwrap_or(self.builder.window_size)
    }

    #[getter]
    fn resolution(&self) -> u64 {
        self.builder.resolution
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /** Returns the sequence indexer for accessing genomic sequences.

        This method provides an indexer that allows access to the genomic sequences
        associated with the dataset. It can be used to retrieve sequences by their keys.

        Returns
        -------
        _SeqIndexer
            An indexer for accessing genomic sequences.
    */
    #[getter]
    fn seq(slf: PyRef<'_, Self>) -> SeqIndexer {
        SeqIndexer(slf.into())
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
        let highlight_start = slf.trim_target.map(|d| d as u64 / slf.builder.resolution);
        let data_indexer = Self::data(slf);
        let key = (region, &tracks).into_pyobject(py)?.into_any();
        let signal_values = data_indexer.__getitem__(py, key)?;
        let track_names = extract_string_list(tracks)?;

        let py_code = r#"
from matplotlib import pyplot as plt
height_per_track = 1.5
width = 8

signal_values = signal_values.T
n_tracks, n_points = signal_values.shape
fig, axes = plt.subplots(n_tracks, 1, figsize=[width, n_tracks * height_per_track], sharex=True)

if n_tracks == 1:
    axes = [axes]

for i, (ax, signal) in enumerate(zip(axes, signal_values)):
    ax.fill_between(range(n_points), 0, signal, color='black')
    if highlight_start is not None:
        ax.axvspan(highlight_start, n_points-highlight_start, color='red', alpha=0.2)
    ax.set_title(track_names[i], fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)

axes[-1].set_xticks([0, n_points - 1])
axes[-1].set_xticklabels(["start", "end"])
axes[-1].set_xlabel(region)

plt.tight_layout()
if savefig is None:
    plt.show()
else:
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
"#;
        py_run!(py, region track_names signal_values highlight_start savefig, py_code);

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

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> DataLoaderIter {
        slf.iter()
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pyclass]
pub struct DataLoaderIter {
    iter: PrefethIterator<(Sequences, ArrayD<f32>)>,
    seq_as_string: bool,
}

impl Iterator for DataLoaderIter {
    type Item = (Sequences, ArrayD<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

#[derive(Debug)]
struct Buffer<T> {
    data: VecDeque<Vec<T>>,
    pos: usize,
    shape: Vec<usize>,
    stride: usize,
}

impl<T: Clone + std::fmt::Debug> Buffer<T> {
    fn new() -> Self {
        Self {
            data: VecDeque::new(),
            pos: 0,
            shape: Vec::new(),
            stride: 1,
        }
    }

    fn remaining(&self) -> usize {
        let n = self.data.iter().map(|d| d.len()).sum::<usize>();
        (n - self.pos) / self.stride
    }

    fn add<D: Dimension>(&mut self, item: Array<T, D>) {
        if self.shape.is_empty() {
            self.shape = item.shape()[1..].to_vec();
            self.stride = self.shape.iter().product();
        }
        assert!(
            item.is_standard_layout(),
            "Buffer only supports standard layout arrays"
        );
        let (data, offset) = item.into_raw_vec_and_offset();
        assert!(
            offset.unwrap_or(0) == 0,
            "Buffer does not support non-zero offset"
        );
        self.data.push_back(data);
    }

    fn take(&mut self, n_record: usize) -> Option<ArrayD<T>> {
        if self.data.is_empty() {
            return None;
        }

        let n = self.stride * n_record;
        let mut result = Vec::with_capacity(n);
        while result.len() < n && !self.data.is_empty() {
            let remaining = self.data[0].len() - self.pos;
            if remaining == 0 {
                self.data.pop_front();
                self.pos = 0;
                continue;
            }

            let take_count = remaining.min(n - result.len());
            result.extend_from_slice(&self.data[0][self.pos..self.pos + take_count]);
            self.pos += take_count;

            if self.pos >= self.data[0].len() {
                self.pos = 0;
                self.data.pop_front();
            }
        }

        if result.is_empty() {
            None
        } else {
            let shape = std::iter::once(result.len() / self.stride)
                .chain(self.shape.iter().cloned())
                .collect::<Vec<_>>();
            Some(ArrayD::from_shape_vec(shape, result).unwrap())
        }
    }

    fn take_all(&mut self) -> ArrayD<T> {
        let n = self.remaining();
        let data: Vec<_> = self.data.drain(..).flatten().skip(self.pos).collect();
        let shape = std::iter::once(n)
            .chain(self.shape.iter().cloned())
            .collect::<Vec<_>>();
        self.pos = 0;
        ArrayD::from_shape_vec(shape, data).unwrap()
    }
}

struct _DataLoaderIter<T> {
    batch_size: usize,
    buffer_seq: Buffer<u8>,
    buffer_data: Buffer<f32>,
    scale: Option<bf16>,
    clamp_max: Option<bf16>,
    chunks: T,
}

impl<T: Iterator<Item = DataChunk>> _DataLoaderIter<T> {
    fn load_chunks(&mut self, n: usize) -> Option<usize> {
        let chunks: Vec<_> = std::iter::repeat_with(|| self.chunks.next())
            .take(n)
            .flatten()
            .collect();
        if chunks.is_empty() {
            return None;
        }

        let data: Vec<_> = chunks
            .into_par_iter()
            .map(|mut chunk| {
                let seqs = chunk.get_seqs().unwrap();
                let mut values = chunk.read_all().unwrap();
                values.transform(self.scale, self.clamp_max);
                (seqs, values)
            })
            .collect();

        let n_read = data.len();
        data.into_iter().for_each(|(seqs, values)| {
            self.buffer_seq.add(seqs.0);
            self.buffer_data.add(values.0.mapv(|x| x.to_f32()));
        });
        Some(n_read)
    }
}

impl<T: Iterator<Item = DataChunk>> Iterator for _DataLoaderIter<T> {
    type Item = (Sequences, ArrayD<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let n_buffer = self.buffer_data.remaining();
        if n_buffer < self.batch_size {
            if let Some(_) = self.load_chunks(4) {
                self.next()
            } else {
                if n_buffer == 0 {
                    None
                } else {
                    let seqs = self
                        .buffer_seq
                        .take_all()
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    let data = self.buffer_data.take_all();
                    Some((Sequences(seqs), data))
                }
            }
        } else {
            let data = self.buffer_data.take(self.batch_size).unwrap();
            let seqs = self
                .buffer_seq
                .take(self.batch_size)
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();
            Some((Sequences(seqs), data))
        }
    }
}

#[pymethods]
impl DataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;
        let values = PyArrayDyn::from_owned_array(py, values);
        let result = if slf.seq_as_string {
            (seq.into_strings(), values).into_pyobject(py).unwrap()
        } else {
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

pub struct PrefethIterator<T>(Arc<Mutex<Receiver<T>>>);

impl<T: Send + 'static> PrefethIterator<T> {
    fn new<I>(iter: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        let (sender, receiver) = sync_channel(buffer_size);
        std::thread::spawn(move || {
            for item in iter {
                sender.send(item).unwrap();
            }
        });
        Self(Arc::new(Mutex::new(receiver)))
    }
}

impl<T: Send + 'static> Iterator for PrefethIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.lock().unwrap().recv().ok()
    }
}

#[pyclass]
pub struct SeqIndexer(Py<GenomeDataLoader>);

impl SeqIndexer {
    fn get(&self, py: Python<'_>, key: &str) -> Result<Array1<u8>> {
        let py_ref = self.0.borrow(py);
        let (chunk, i) = py_ref
            .builder
            .seq_index
            .get(&GenomicRange::from_str(key).unwrap())
            .with_context(|| format!("Failed to get data chunk for key: {}", key))?;
        Ok(chunk.open(false)?.get_seq_at(*i)?)
    }
}

#[pymethods]
impl SeqIndexer {
    fn __getitem__<'a>(
        &'a self,
        py: Python<'a>,
        key: Bound<'a, PyAny>,
    ) -> Result<Bound<'a, PyAny>> {
        if let Ok(key) = key.extract::<String>() {
            Ok(PyArray1::from_owned_array(py, self.get(py, &key)?).into_any())
        } else {
            let key: Vec<String> = key.extract()?;
            todo!()
        }
    }
}

#[pyclass]
pub struct DataIndexer(Py<GenomeDataLoader>);

impl DataIndexer {
    fn get(&self, py: Python<'_>, key: &str, j: &[String]) -> Result<ArrayD<bf16>> {
        let py_ref = self.0.borrow(py);
        let (chunk, i) = py_ref
            .builder
            .seq_index
            .get(&GenomicRange::from_str(key).unwrap())
            .with_context(|| format!("Failed to get data chunk for key: {}", key))?;
        let vals = chunk.open(false)?.gets(j)?;
        Ok(vals
            .0
            .axis_iter(ndarray::Axis(0))
            .nth(*i)
            .unwrap()
            .to_owned())
    }
}

#[pymethods]
impl DataIndexer {
    fn __getitem__<'a>(
        &'a self,
        py: Python<'a>,
        key: Bound<'a, PyAny>,
    ) -> Result<Bound<'a, PyArrayDyn<f32>>> {
        let (i, j): (String, Bound<'_, PyAny>) = key.extract()?;
        let j = if let Ok(j_) = j.extract::<String>() {
            vec![j_]
        } else {
            j.extract::<Vec<String>>()?
        };

        let data = self.get(py, &i, &j)?.mapv(|x| x.to_f32());
        Ok(PyArrayDyn::from_owned_array(py, data))
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
    trim_target: Optional[int]
        Optional parameter to specify the trim target for all loaders.
        If not provided, each loader will use its own trim target.
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
            loaders, *, batch_size=None, trim_target=None, window_size=None, seq_as_string=false,
        ),
        text_signature = "($self, loaders, *, batch_size=None, trim_target=None, window_size=None, seq_as_string=False)"
    )]
    pub fn new(
        mut loaders: IndexMap<String, GenomeDataLoader>,
        batch_size: Option<usize>,
        trim_target: Option<usize>,
        window_size: Option<u64>,
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
            trim_target.map(|t| {
                loader.set_trim_target(t);
            });
            window_size.map(|w| {
                loader.window_size = Some(w);
            });
        });

        ensure!(
            loaders
                .values()
                .map(|loader| loader.window_size)
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
                let n_tracks = v.builder.tracks().unwrap().len();
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
        self.0[0]
            .builder
            .seq_index
            .keys()
            .map(|x| x.pretty_show())
            .collect()
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
pub struct MultiDataLoaderIter(IndexMap<String, DataLoaderIter>);

impl Iterator for MultiDataLoaderIter {
    type Item = (Sequences, IndexMap<String, ArrayD<f32>>);

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
            .map(|(tag, v)| (tag, PyArrayDyn::from_owned_array(py, v)))
            .collect::<IndexMap<_, _>>();

        let result = if slf.0[0].seq_as_string {
            (seq.into_strings(), values).into_pyobject(py).unwrap()
        } else {
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

/** This class combines multiple `GenomeDataLoaderMap` instances into a single loader.

    It allows for the simultaneous loading of genomic data from multiple species.

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
    type Item = (Sequences, IndexMap<String, ArrayD<f32>>);

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
            .map(|(tag, v)| (tag, PyArrayDyn::from_owned_array(py, v)))
            .collect::<IndexMap<_, _>>();

        let result = if slf.iters[0].0[0].seq_as_string {
            (seq.into_strings(), values).into_pyobject(py).unwrap()
        } else {
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
