use anyhow::Result;
use bed_utils::bed::{map::GIntervalMap, GenomicRange};
use half::bf16;
use itertools::Itertools;
use ndarray::Array3;
use rand::{seq::SliceRandom, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::dataloader::chunk::{DataChunk, Sequences};

#[derive(Debug, Clone)]
pub(crate) struct ReadChunkOptions {
    pub write: bool,
    pub split_data: Option<(usize, usize)>, // Optional size for splitting data, 1st is for sequences, 2nd for values
    pub trim_target: Option<usize>,         // Optional output trimming target
    pub aggregation: Option<usize>,         // Optional aggregation size
    pub scale: Option<f32>,
    pub clamp_max: Option<f32>, 
}

impl Default for ReadChunkOptions {
    fn default() -> Self {
        Self {
            write: false,
            split_data: None,
            trim_target: None,
            aggregation: None,
            scale: None,
            clamp_max: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ChunkInfo {
    chrom: String,
    id: usize,
    path: PathBuf,
}

impl ChunkInfo {
    pub fn open(&self, opts: &ReadChunkOptions) -> Result<DataChunk> {
        let mut chunk = DataChunk::open(&self.path, opts.write)?;
        if let Some(trim_target) = opts.trim_target {
            chunk.set_trim_target(trim_target);
        }
        if let Some(split_data) = opts.split_data {
            chunk.set_split_data(split_data);
        }
        if let Some(aggregation) = opts.aggregation {
            chunk.set_aggregation(aggregation);
        }
        Ok(chunk)   
    }
}

#[derive(Debug, Clone)]
pub struct ChunkIndex(pub(crate) BTreeMap<GenomicRange, (ChunkInfo, usize)>);

impl ChunkIndex {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get_chunk_size(&self) -> usize {
        let chunk_info = &self.0.values().next().unwrap().0;
        chunk_info.open(&Default::default()).unwrap().len()
    }

    /// Returns all chromosomes in the index.
    pub fn chromosomes(&self) -> impl Iterator<Item = &String> {
        self.0
            .values()
            .map(|(chunk_info, _)| &chunk_info.chrom)
            .unique()
    }

    pub fn keys(&self) -> impl Iterator<Item = &GenomicRange> {
        self.0.keys()
    }

    pub fn get(&self, key: &GenomicRange) -> Option<&(ChunkInfo, usize)> {
        self.0.get(key)
    }

    /// Compute the intersection of the index with the given genomic regions.
    pub fn intersection(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let interval_map: GIntervalMap<_> =
            self.0.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
        let index = regions
            .flat_map(|region| interval_map.find(&region))
            .unique_by(|x| x.0.clone())
            .map(|(a, b)| (a, b.clone()))
            .collect();
        Self(index)
    }

    /// Compute the difference of the index with the given genomic regions.
    pub fn difference(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let mut new_index = self.0.clone();
        let interval_map: GIntervalMap<_> =
            self.0.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
        regions.for_each(|region| {
            interval_map.find(&region).for_each(|(a, _)| {
                new_index.remove(&a);
            })
        });
        Self(new_index)
    }

    pub fn iter_chunks<R: Rng>(
        &self,
        opts: ReadChunkOptions,
        shuffle: Option<&mut R>,
    ) -> impl Iterator<Item = DataChunk> {
        let mut chunks: Vec<_> = self
            .0
            .values()
            .sorted()
            .chunk_by(|x| x.0.clone())
            .into_iter()
            .map(|(chunk, group)| {
                let idx = group.into_iter().map(|(_, i)| *i).collect::<Vec<_>>();
                (chunk, idx)
            })
            .collect();
        if let Some(rng) = shuffle {
            chunks.shuffle(rng);
        }
        chunks.into_iter().map(move |(chunk, idx)| {
            let mut c = chunk.open(&opts).unwrap();
            c.subset(idx).unwrap();
            c
        })
    }

    pub fn iter_chunk_data<R: Rng>(
        &self,
        opts: ReadChunkOptions,
        shuffle: Option<&mut R>,
        prefetch: usize,
    ) -> impl Iterator<Item = (Sequences, Array3<f32>)> {
        let mut chunks: Vec<_> = self
            .0
            .values()
            .sorted()
            .chunk_by(|x| x.0.clone())
            .into_iter()
            .map(|(chunk, group)| {
                let idx = group.into_iter().map(|(_, i)| *i).collect::<Vec<_>>();
                (chunk, idx)
            })
            .collect();
        if let Some(rng) = shuffle {
            chunks.shuffle(rng);
        }
        let chunks: Vec<_> = chunks.into_iter().chunks(prefetch).into_iter().map(|x| x.collect::<Vec<_>>()).collect();
        let scale = opts.scale.map(bf16::from_f32);
        let clamp_max = opts.clamp_max.map(bf16::from_f32);
        chunks.into_iter().flat_map(move |group| {
            let opts = opts.clone();
            group.into_par_iter().map(move |(chunk, idx)| {
                let mut chunk= chunk.open(&opts).unwrap();
                chunk.subset(idx).unwrap();
                let seqs = chunk.get_seqs().unwrap();
                let mut values = chunk.read_all();
                values.transform(scale, clamp_max);
                (seqs, values.0.mapv(|x| x.to_f32()))
            }).collect::<Vec<_>>()
        })
    }
}

pub(crate) fn make_seq_index<'a>(
    location: impl AsRef<Path>,
    chroms: impl Iterator<Item = &'a String>,
) -> ChunkIndex {
    let index = chroms
        .flat_map(|chr| {
            std::fs::read_dir(location.as_ref().join(&chr))
                .unwrap()
                .flat_map(|entry| {
                    let entry = entry.unwrap();
                    let chunk = DataChunk::open(&entry.path(), false).unwrap();
                    chunk
                        .segments
                        .iter()
                        .enumerate()
                        .map(|(i, segment)| {
                            let item = ChunkInfo {
                                path: entry.path(),
                                chrom: chr.clone(),
                                id: entry
                                    .file_name()
                                    .to_str()
                                    .unwrap()
                                    .parse::<usize>()
                                    .unwrap(),
                            };
                            (segment.clone(), (item, i))
                        })
                        .collect::<Vec<_>>()
                })
        })
        .collect();
    ChunkIndex(index)
}
