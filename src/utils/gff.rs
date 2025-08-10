//! # Genomic Feature Counter Module
//!
//! This module provides the functionality to count genomic features (such as genes or transcripts)
//! in genomic data. The primary structures in this module are `TranscriptCount` and `GeneCount`,
//! both of which implement the `FeatureCounter` trait. The `FeatureCounter` trait provides a
//! common interface for handling feature counts, including methods for resetting counts,
//! updating counts, and retrieving feature IDs, names, and counts.
//!
//! `SparseCoverage`, from the bed_utils crate, is used for maintaining counts of genomic features,
//! and this structure also implements the `FeatureCounter` trait in this module.
//!
//! `TranscriptCount` and `GeneCount` structures also hold a reference to `Promoters`, which
//! provides additional information about the genomic features being counted.
//!
//! To handle the mapping of gene names to indices, an `IndexMap` is used in the `GeneCount` structure.
//! This allows for efficient look-up of gene indices by name, which is useful when summarizing counts
//! at the gene level.
//!
//! The module aims to provide a comprehensive, efficient, and flexible way to handle and manipulate
//! genomic feature counts in Rust.
use pyo3::prelude::*;
use anyhow::{bail, Context, Result};
use noodles::{core::Position, gff::{self, feature::record::Strand}};
use noodles::gff::feature::Record;
use std::{fmt::Debug, io::BufReader, path::PathBuf};

struct ParserOptions {
    transcript_name_key: String,
    transcript_id_key: String,
    gene_name_key: String,
    gene_id_key: String,
}

impl<'a> Default for ParserOptions {
    fn default() -> Self {
        Self {
            transcript_name_key: "transcript_name".to_string(),
            transcript_id_key: "transcript_id".to_string(),
            gene_name_key: "gene_name".to_string(),
            gene_id_key: "gene_id".to_string(),
        }
    }
}

/// Position is 0-based.
#[pyclass]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gene {
    #[pyo3(get)]
    pub gene_name: String,
    #[pyo3(get)]
    pub gene_id: String,
    #[pyo3(get)]
    pub is_coding: Option<bool>,
    #[pyo3(get)]
    pub chrom: String,
    pub left: Position,
    pub right: Position,
    pub strand: Strand,
}

impl Gene {
    fn from_gff<R: Record + Debug>(record: &R, options: &ParserOptions) -> Result<Self> {
        if record.ty() != "gene" {
            bail!("record is not a gene");
        }

        let left = record.feature_start()?;
        let right = record.feature_end()?;
        let attributes = record.attributes();
        let get_attr = |key: &str| -> String {
            attributes
                .get(key.as_bytes())
                .expect(&format!("failed to find '{}' in record: {:?}", key, record)).unwrap()
                .as_string().unwrap().to_string()
        };
        let get_attr_maybe = |key: &str| -> Option<String> {
            attributes
                .get(key.as_bytes())
                .map(|v| v.unwrap().as_string().unwrap().to_string())
        };

        Ok(Self {
            gene_name: get_attr(options.gene_name_key.as_str()),
            gene_id: get_attr(options.gene_id_key.as_str()),
            is_coding: get_attr_maybe("gene_type").map(|x| x == "protein_coding"),
            chrom: record.reference_sequence_name().to_string(),
            left,
            right,
            strand: record.strand()?,
        })
    }
}

#[pymethods]
impl Gene {
    #[getter]
    fn get_start(&self) -> usize {
        self.left.get() - 1
    }
    
    #[getter]
    fn get_end(&self) -> usize {
        self.right.get()
    }

    #[getter]
    fn get_strand(&self) -> &str {
        if self.strand == Strand::Forward {
            "+"
        } else if self.strand == Strand::Reverse {
            "-"
        } else {
            panic!("Strand is not set")
        }
    }

    fn get_tss(&self) -> Option<usize> {
        match self.strand {
            Strand::Forward => Some(<Position as TryInto<usize>>::try_into(self.left).unwrap() - 1),
            Strand::Reverse => {
                Some(<Position as TryInto<usize>>::try_into(self.right).unwrap() - 1)
            }
            _ => None,
        }
    }
}


/// Position is 0-based.
#[pyclass]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transcript {
    #[pyo3(get)]
    pub transcript_name: Option<String>,
    #[pyo3(get)]
    pub transcript_id: String,
    #[pyo3(get)]
    pub gene_name: String,
    #[pyo3(get)]
    pub gene_id: String,
    #[pyo3(get)]
    pub is_coding: Option<bool>,
    #[pyo3(get)]
    pub chrom: String,
    pub left: Position,
    pub right: Position,
    pub strand: Strand,
}

impl Transcript {
    fn from_gff<R: Record + Debug>(record: &R, options: &ParserOptions) -> Result<Self> {
        if record.ty() != "transcript" {
            bail!("record is not a transcript");
        }

        let left = record.feature_start()?;
        let right = record.feature_end()?;
        let attributes = record.attributes();
        let get_attr = |key: &str| -> String {
            attributes
                .get(key.as_bytes())
                .expect(&format!("failed to find '{}' in record: {:?}", key, record)).unwrap()
                .as_string().unwrap().to_string()
        };
        let get_attr_maybe = |key: &str| -> Option<String> {
            attributes
                .get(key.as_bytes())
                .map(|v| v.unwrap().as_string().unwrap().to_string())
        };

        Ok(Transcript {
            transcript_name: get_attr_maybe(options.transcript_name_key.as_str()),
            transcript_id: get_attr(options.transcript_id_key.as_str()),
            gene_name: get_attr(options.gene_name_key.as_str()),
            gene_id: get_attr(options.gene_id_key.as_str()),
            is_coding: get_attr_maybe("transcript_type").map(|x| x == "protein_coding"),
            chrom: record.reference_sequence_name().to_string(),
            left,
            right,
            strand: record.strand()?,
        })
    }
}

#[pymethods]
impl Transcript {
    #[getter]
    fn get_start(&self) -> usize {
        self.left.get() - 1
    }
    
    #[getter]
    fn get_end(&self) -> usize {
        self.right.get()
    }

    #[getter]
    fn get_strand(&self) -> &str {
        if self.strand == Strand::Forward {
            "+"
        } else if self.strand == Strand::Reverse {
            "-"
        } else {
            panic!("Strand is not set")
        }
    }

    fn get_tss(&self) -> Option<usize> {
        match self.strand {
            Strand::Forward => Some(<Position as TryInto<usize>>::try_into(self.left).unwrap() - 1),
            Strand::Reverse => {
                Some(<Position as TryInto<usize>>::try_into(self.right).unwrap() - 1)
            }
            _ => None,
        }
    }
}


#[pyfunction]
#[pyo3(
    signature = (gff_file, *,
        transcript_name_key="transcript_name", transcript_id_key="transcript_id",
        gene_name_key="gene_name", gene_id_key="gene_id"
    ),
    text_signature = "(gff_file, *,
        transcript_name_key='transcript_name', transcript_id_key='transcript_id',
        gene_name_key='gene_name', gene_id_key='gene_id')"
)]
pub fn read_transcripts(
    gff_file: PathBuf,
    transcript_name_key: &str,
    transcript_id_key: &str,
    gene_name_key: &str,
    gene_id_key: &str,
) -> Result<Vec<Transcript>>
{
    let opts = ParserOptions {
        transcript_name_key: transcript_name_key.to_string(),
        transcript_id_key: transcript_id_key.to_string(),
        gene_name_key: gene_name_key.to_string(),
        gene_id_key: gene_id_key.to_string(),
    };
    let reader = std::fs::File::open(&gff_file)
        .with_context(|| format!("failed to open GFF file: {:?}", gff_file))?;
    let reader = flate2::read::MultiGzDecoder::new(reader);
    let mut reader = gff::io::Reader::new(BufReader::new(reader));
    let mut results = Vec::new();
    for record in reader.record_bufs() {
        let rec = record.with_context(|| "failed to read GFF record")?;
        if rec.ty() == "transcript" {
            results.push(Transcript::from_gff(&rec, &opts)?);
        }
    }
    Ok(results)
}


#[pyfunction]
#[pyo3(
    signature = (gff_file, *, gene_name_key="gene_name", gene_id_key="gene_id"),
    text_signature = "(gff_file, *, gene_name_key='gene_name', gene_id_key='gene_id')"
)]
pub fn read_genes(
    gff_file: PathBuf,
    gene_name_key: &str,
    gene_id_key: &str,
) -> Result<Vec<Gene>>
{
    let opts = ParserOptions {
        gene_name_key: gene_name_key.to_string(),
        gene_id_key: gene_id_key.to_string(),
        ..Default::default()
    };
    let reader = std::fs::File::open(&gff_file)
        .with_context(|| format!("failed to open GFF file: {:?}", gff_file))?;
    let reader = flate2::read::MultiGzDecoder::new(reader);
    let mut reader = gff::io::Reader::new(BufReader::new(reader));
    let mut results = Vec::new();
    for record in reader.record_bufs() {
        let rec = record.with_context(|| "failed to read GFF record")?;
        if rec.ty() == "gene" {
            results.push(Gene::from_gff(&rec, &opts)?);
        }
    }
    Ok(results)
}