use std::{collections::BTreeMap, path::{Path, PathBuf}};
use crate::dataloader::chunk::DataChunk;

pub struct SegmentIndex(BTreeMap<String, (PathBuf, usize)>);

impl SegmentIndex {
    pub fn new<'a>(location: impl AsRef<Path>, chroms: impl Iterator<Item = &'a String>) -> Self {
        let index = chroms.flat_map(|chr| {
            std::fs::read_dir(location.as_ref().join(&chr)).unwrap().flat_map(|entry| {
                let path = entry.unwrap().path();
                let chunk = DataChunk::open(&path).unwrap();
                chunk.segments.iter().enumerate().map(|(i, segment)| {
                    (segment.pretty_show(), (path.clone(), i))
                }).collect::<Vec<_>>()
            })
        }).collect();
        SegmentIndex(index)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn segments(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub fn get_datachunk(&self, key: &str) -> Option<(DataChunk, usize)> {
        let (path, i) = self.0.get(key)?;
        let chunk = DataChunk::open(path).unwrap();
        Some((chunk, *i))
    }
}