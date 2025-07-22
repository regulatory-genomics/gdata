use anyhow::{ensure, Context, Result};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayD};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::{prelude::*, py_run};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use crate::dataloader::builder::GenomeDataBuilder;
use crate::dataloader::chunk::{DataChunk, Sequences, Values};
use crate::utils::PrefetchIterator;

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
    seq_as_string : bool
        If True, sequences will be returned as strings instead of numpy integer arrays.
        This is useful for cases where you want to work with the sequences as text,
        such as for visualization or text-based analysis.
    prefetch : int
        The number of chunks to prefetch for efficient data loading (default is 1).
        This allows for asynchronous loading of data, improving performance during training or inference.
        But it will increase memory usage, so it should be set according to the available resources.

    See Also
    --------
    GenomeDataBuilder
    MultiGenomeDataLoader

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
pub struct GenomeDataLoader {
    builder: Arc<GenomeDataBuilder>,
    batch_size: usize,
    trim_target: Option<usize>,
    seq_as_string: bool,
    prefetch: usize,
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
            self.builder.window_size,
            self.builder.resolution,
            self.batch_size,
            self.trim_target.unwrap_or(0),
        )?;
        Ok(())
    }
}

impl GenomeDataLoader {
    fn iter(&self) -> DataLoaderIter {
        DataLoaderIter {
            batch_size: self.batch_size,
            buffer_seq: VecDeque::new(),
            buffer_data: VecDeque::new(),
            chunks: self
                .builder
                .iter_chunk_data(self.prefetch, self.trim_target, false),
            seq_as_string: self.seq_as_string,
        }
    }
}

#[pymethods]
impl GenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (
            location, *, batch_size=8, trim_target=None, seq_as_string=false, prefetch=1,
        ),
        text_signature = "($self, location, *, batch_size=8, trim_target=None, seq_as_string=False, prefetch=1)"
    )]
    pub fn new(
        location: PathBuf,
        batch_size: usize,
        trim_target: Option<usize>,
        seq_as_string: bool,
        prefetch: usize,
    ) -> Result<Self> {
        Ok(Self {
            builder: Arc::new(GenomeDataBuilder::open(location)?),
            batch_size,
            trim_target,
            seq_as_string,
            prefetch,
        })
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
    fn segments(&self) -> Vec<&String> {
        self.builder.seq_index.keys().collect()
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
    fn seq(&self) -> SeqIndexer {
        SeqIndexer(self.builder.clone())
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
    fn data(&self) -> DataIndexer {
        DataIndexer(self.builder.clone())
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
        &self,
        py: Python<'_>,
        region: &str,
        tracks: Bound<'_, PyAny>,
        savefig: Option<PathBuf>,
    ) -> Result<()> {
        let data_indexer = self.data();
        let key = (region, &tracks).into_pyobject(py)?.into_any();
        let signal_values = data_indexer.__getitem__(py, key)?;
        let track_names = extract_string_list(tracks)?;
        let highlight_start = self.trim_target.map(|d| d as u64 / self.builder.resolution);

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
    #ax.set_ylabel('signal', rotation=0, labelpad=20, va='center', ha='right')
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

    fn __len__(&self) -> usize {
        let n = self.builder.seq_index.len();
        n / self.batch_size + if n % self.batch_size > 0 { 1 } else { 0 }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> DataLoaderIter {
        slf.iter()
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pyclass]
pub struct DataLoaderIter {
    batch_size: usize,
    buffer_seq: VecDeque<Array1<u8>>,
    buffer_data: VecDeque<ArrayD<f32>>,
    chunks: PrefetchIterator<(Sequences, Values)>,
    seq_as_string: bool,
}

impl Iterator for DataLoaderIter {
    type Item = (Sequences, ArrayD<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let n_buffer = self.buffer_seq.len();
        if n_buffer < self.batch_size {
            if let Some((seqs, values)) = self.chunks.next() {
                self.buffer_seq.extend(seqs.iter_rows());
                self.buffer_data.extend(values.iter_rows());
                self.next()
            } else {
                if n_buffer == 0 {
                    None
                } else {
                    let r = self.buffer_seq.len();
                    let c = self.buffer_seq[0].len();
                    let seqs = self.buffer_seq.drain(..).flatten().collect::<Vec<_>>();
                    let seqs = Array2::from_shape_vec((r, c), seqs).unwrap();
                    let mut shape: Vec<_> = self.buffer_data[0].shape().to_vec();
                    shape.insert(0, r);
                    let data = self.buffer_data.drain(..).flatten().collect::<Vec<_>>();
                    let data = ArrayD::from_shape_vec(shape, data).unwrap();
                    Some((Sequences(seqs), data))
                }
            }
        } else {
            let c = self.buffer_seq[0].len();
            let seqs = self
                .buffer_seq
                .drain(..self.batch_size)
                .flatten()
                .collect::<Vec<_>>();
            let seqs = Array2::from_shape_vec((self.batch_size, c), seqs).unwrap();
            let mut shape = self.buffer_data[0].shape().to_vec();
            shape.insert(0, self.batch_size);
            let data = self
                .buffer_data
                .drain(..self.batch_size)
                .flatten()
                .collect::<Vec<_>>();
            let data = ArrayD::from_shape_vec(shape, data).unwrap();
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

#[pyclass]
pub struct SeqIndexer(Arc<GenomeDataBuilder>);

impl SeqIndexer {
    fn get(&self, key: &str) -> Result<Array1<u8>> {
        let (chunk, i) = self
            .0
            .seq_index
            .get(key)
            .with_context(|| format!("Failed to get data chunk for key: {}", key))?;
        Ok(DataChunk::open(chunk, false)?.get_seq_at(*i)?)
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
            Ok(PyArray1::from_owned_array(py, self.get(&key)?).into_any())
        } else {
            let key: Vec<String> = key.extract()?;
            todo!()
        }
    }
}

#[pyclass]
pub struct DataIndexer(Arc<GenomeDataBuilder>);

impl DataIndexer {
    fn get(&self, key: &str, j: &[String]) -> Result<ArrayD<bf16>> {
        let (chunk, i) = self
            .0
            .seq_index
            .get(key)
            .with_context(|| format!("Failed to get data chunk for key: {}", key))?;
        let vals = DataChunk::open(chunk, false)?.gets(j)?;
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

        let data = self.get(&i, &j)?.mapv(|x| x.to_f32());
        Ok(PyArrayDyn::from_owned_array(py, data))
    }
}

/** Similar to `GenomeDataLoader`, but for multiple genomic datasets.
    
    This class allows you to load and iterate over multiple genomic datasets simultaneously,
    each identified by a unique tag. It ensures that all datasets share the same genomic segments,
    which is a requirement for consistent data processing.

    Parameters
    ----------
    locations : dict[str, Path]
        A dictionary mapping tags to paths of genomic data directories.
        Each path should point to a directory containing genomic data files.
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
    seq_as_string : bool
        If True, sequences will be returned as strings instead of numpy integer arrays.
        This is useful for cases where you want to work with the sequences as text,
        such as for visualization or text-based analysis.
    prefetch : int
        The number of chunks to prefetch for efficient data loading (default is 1).
        This allows for asynchronous loading of data, improving performance during training or inference.
        But it will increase memory usage, so it should be set according to the available resources.

    See Also
    --------
    GenomeDataLoader
    GenomeDataBuilder

    Examples
    --------
    >>> from gdata import as MultiGenomeDataLoader
    >>> loader = MultiGenomeDataLoader({'ChIP': "data1", 'DNase': "data2"}, trim_target=40_960)
    >>> for seq, data in loader:
    ...     print(seq, data)
*/
#[pyclass]
pub struct MultiGenomeDataLoader(IndexMap<String, GenomeDataLoader>);

impl std::fmt::Display for MultiGenomeDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(
            f,
            "MultiGenomeDataLoader with keys: {}",
            self.0.keys().map(|x| "'".to_owned() + x + "'").join(", "),
        )?;
        Ok(())
    }
}

#[pymethods]
impl MultiGenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (
            locations, *, batch_size=8, trim_target=None, seq_as_string=false, prefetch=1,
        ),
        text_signature = "($self, locations, *, batch_size=8, trim_target=None, seq_as_string=False, prefetch=1)"
    )]
    pub fn new(
        locations: IndexMap<String, PathBuf>,
        batch_size: usize,
        trim_target: Option<usize>,
        seq_as_string: bool,
        prefetch: usize,
    ) -> Result<Self> {
        let loaders = locations.into_iter()
            .map(|(tag, loc)| {
                let loader = GenomeDataLoader::new(
                    loc,
                    batch_size,
                    trim_target,
                    seq_as_string,
                    prefetch,
                )?;
                Ok((tag, loader))
            })
            .collect::<Result<IndexMap<_, _>>>()?;
        ensure!(
            loaders.values().map(|loader| loader.segments()).all_equal(),
            "All genome data loaders must have the same segments"
        );
        Ok(Self(loaders))
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    #[getter]
    fn segments(&self) -> Vec<&String> {
        self.0[0].builder.seq_index.keys().collect()
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.0[0].batch_size
    }

    fn __iter__(slf: PyRef<'_, Self>) -> MultiDataLoaderIter {
        let iter = slf
            .0
            .iter()
            .map(|(tag, loader)| (tag.clone(), loader.iter()))
            .collect();
        MultiDataLoaderIter(iter)
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

fn extract_string_list(str: Bound<'_, PyAny>) -> Result<Vec<String>> {
    if let Ok(s) = str.extract::<String>() {
        Ok(vec![s])
    } else {
        str.extract::<Vec<String>>()
            .with_context(|| "Failed to extract string list from PyAny")
    }
}
