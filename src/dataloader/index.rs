use anyhow::Result;
use bed_utils::bed::{map::GIntervalMap, GenomicRange};
use itertools::Itertools;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::dataloader::chunk::DataChunk;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ChunkInfo {
    chrom: String,
    id: usize,
    path: PathBuf,
}

impl ChunkInfo {
    pub fn open(&self, write: bool) -> Result<DataChunk> {
        DataChunk::open(&self.path, write)
    }
}

#[derive(Debug, Clone)]
pub struct ChunkIndex(BTreeMap<GenomicRange, (ChunkInfo, usize)>);

impl ChunkIndex {
    pub fn len(&self) -> usize {
        self.0.len()
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

    pub fn intersect(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let interval_map: GIntervalMap<_> =
            self.0.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
        let index = regions
            .flat_map(|region| interval_map.find(&region))
            .unique_by(|x| x.0.clone())
            .map(|(a, b)| (a, b.clone()))
            .collect();
        Self(index)
    }

    pub fn iter_chunks(
        &self,
        trim_target: Option<usize>,
        write: bool,
    ) -> impl Iterator<Item = DataChunk> {
        let chunks: Vec<_> = self
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
        chunks.into_iter().map(move |(chunk, idx)| {
            let mut chunk = chunk.open(write).unwrap();
            chunk.subset(idx).unwrap();
            if let Some(trim_target) = trim_target {
                chunk.set_trim_target(trim_target);
            }
            chunk
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
