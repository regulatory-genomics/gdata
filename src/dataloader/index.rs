use std::{collections::BTreeMap, path::{Path, PathBuf}};
use crate::dataloader::chunk::DataChunk;

pub struct ChunkIndex(BTreeMap<String, (PathBuf, usize)>);

impl ChunkIndex {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub fn get(&self, key: &str) -> Option<&(PathBuf, usize)> {
        self.0.get(key)
    }
}

pub(crate) fn make_seq_index<'a>(location: impl AsRef<Path>, chroms: impl Iterator<Item = &'a String>) -> ChunkIndex {
    let index = chroms.flat_map(|chr| {
        std::fs::read_dir(location.as_ref().join(&chr)).unwrap().flat_map(|entry| {
            let path = entry.unwrap().path();
            let chunk = DataChunk::open(&path).unwrap();
            chunk.segments.iter().enumerate().map(|(i, segment)| {
                (segment.pretty_show(), (path.clone(), i))
            }).collect::<Vec<_>>()
        })
    }).collect();
    ChunkIndex(index)
}