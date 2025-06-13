use std::path::Path;
use std::{io::Read, path::PathBuf};
use bed_utils::bed::map::GIntervalMap;
use bed_utils::bed::{BEDLike, BedGraph, MergeBed, SortBed, Strand, BED};
use bigtools::BigWigWrite;
use bstr::BString;
use itertools::Itertools;
use noodles::bam;
use anyhow::Result;
use noodles::sam;
use pyo3::prelude::*;
use noodles::sam::header::{Header, ReferenceSequences};
use noodles::sam::alignment::record::cigar::op::Kind;
use noodles::sam::alignment::record::cigar::Op;
use noodles::sam::alignment::record_buf::Cigar;

#[derive(Debug)]
struct Stats {
    n_total: usize,
    n_unmapped: usize,
    n_secondary: usize,
    n_supplementary: usize,
    n_paired: usize,
    n_spliced: usize,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            n_total: 0,
            n_unmapped: 0,
            n_secondary: 0,
            n_supplementary: 0,
            n_paired: 0,
            n_spliced: 0,
        }
    }
}

/// SpliceSegment represents a contiguous block of cigar operations not containing
/// any "Skip" sections. The SpliceSegment is 0-based, half-open with respect to the reference.
#[derive(Debug)]
struct SpliceSegment {
    start: u64,
    end: u64,
    cigar: Cigar,
}

/// SpliceSegments is used to represent the alignment of a read to a transcript.
/// It consists of the left and right clipping operations, and a list of SpliceSegments.
pub struct SpliceSegments {
    left_clip: Cigar,
    right_clip: Cigar,
    segments: Vec<SpliceSegment>,
}

impl SpliceSegments {
    fn new<R: sam::alignment::Record>(read: &R) -> Result<Self> {
        let cigar = read.cigar();
        let alignment_start = read.alignment_start().unwrap().unwrap().get();

        let mut left_clip: Vec<Op> = Vec::new();
        let mut right_clip: Vec<Op> = Vec::new();
        let mut splice_segments: Vec<SpliceSegment> = Vec::new();
        let mut seen_nonclips = false; // whether we've seen non-clip bases yet
        let mut curr_segment = SpliceSegment {
            start: alignment_start as u64,
            end: alignment_start as u64,
            cigar: Vec::new().into(),
        };

        for c in cigar.as_ref() {
            let c = c?;
            match c.kind() {
                Kind::HardClip | Kind::SoftClip => {
                    if seen_nonclips {
                        right_clip.push(c);
                    } else {
                        left_clip.push(c);
                    }
                }
                Kind::Skip => {
                    seen_nonclips = true;
                    let next_start = curr_segment.end + c.len() as u64;
                    splice_segments.push(curr_segment);
                    curr_segment = SpliceSegment {
                        start: next_start,
                        end: next_start,
                        cigar: Vec::new().into(),
                    };
                }
                Kind::Insertion => {
                    seen_nonclips = true;
                    curr_segment.cigar.as_mut().push(c);
                }
                Kind::Match | Kind::Deletion | Kind::SequenceMatch | Kind::SequenceMismatch => {
                    seen_nonclips = true;
                    curr_segment.end += c.len() as u64;
                    curr_segment.cigar.as_mut().push(c);
                }
                Kind::Pad => unreachable!(),
            }
        }
        splice_segments.push(curr_segment);

        Ok(Self {
            left_clip: left_clip.into(),
            right_clip: right_clip.into(),
            segments: splice_segments,
        })
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

fn read_malformed_header<R: std::io::Read>(reader: &mut bam::io::reader::Reader<R>) -> Result<Header> {
    let mut header_reader = reader.header_reader();
    header_reader.read_magic_number()?;
    let mut raw_sam_header_reader = header_reader.raw_sam_header_reader()?;
    let mut raw_header = String::new();
    raw_sam_header_reader.read_to_string(&mut raw_header)?;
    raw_sam_header_reader.discard_to_end()?;

    Ok(Header::builder().set_reference_sequences(header_reader.read_reference_sequences()?).build())
}

#[pyfunction]
pub fn bam_cov(input: PathBuf, output: PathBuf) -> Result<()> {
    let mut stats = Stats::new();

    let mut reader = bam::io::reader::Builder::default().build_from_path(input)?;
    let header = read_malformed_header(&mut reader)?;
    
    let fragments = reader.records().flat_map(|record| {
        let record = record.unwrap();
        stats.n_total += 1;

        let flag = record.flags();
        if flag.is_segmented() {
            stats.n_paired += 1;
        }
        if flag.is_unmapped() {
            stats.n_unmapped += 1;
            None
        } else {
            if flag.is_secondary() {
                stats.n_secondary += 1;
                None
            } else if flag.is_supplementary() {
                stats.n_supplementary += 1;
                None
            } else {
                let spliced_segments = SpliceSegments::new(&record).unwrap();
                if spliced_segments.num_segments() > 1 {
                    stats.n_spliced += 1;
                }
                let ref_id = record.reference_sequence_id().unwrap().unwrap();
                let chrom = header.reference_sequences().get_index(ref_id).unwrap().0.to_string();
                let strand = if flag.is_reverse_complemented() {
                    Strand::Reverse
                } else {
                    Strand::Forward
                };
                let beds = spliced_segments.segments.into_iter().map(move |segment| {
                    BED::<6>::new(
                        &chrom, 
                        segment.start, 
                        segment.end, 
                        None,
                        None,
                        Some(strand.clone()),
                        Default::default(),
                    )
                }).collect::<Vec<_>>();
                Some(beds)
            }
        }
    }).flatten().sort_bed().unwrap().map(|x| x.unwrap());

    let bedgraph = create_bedgraph_from_sorted_fragments(fragments, header.reference_sequences(), 1, None, None, Some(Normalization::RPKM), None, None);
    create_bigwig_from_bedgraph(bedgraph.into_iter(), header.reference_sequences(), output)?;
    Ok(())
}

/// Create a BedGraph file from fragments.
///
/// The values represent the sequence coverage (or sequencing depth), which refers
/// to the number of reads that include a specific nucleotide of a reference genome.
/// For paired-end data, the coverage is computed as the number of times a base
/// is read or spanned by paired ends or mate paired reads
///
/// # Arguments
///
/// * `fragments` - iterator of fragments
/// * `chrom_sizes` - chromosome sizes
/// * `bin_size` - Size of the bins, in bases, for the output of the bigwig/bedgraph file.
/// * `smooth_base` - Length of the smoothing base. If None, no smoothing is performed.
/// * `blacklist_regions` - Blacklist regions to be ignored.
/// * `normalization` - Normalization method.
/// * `include_for_norm` - If specified, only the regions that overlap with these intervals will be used for normalization.
/// * `exclude_for_norm` - If specified, the regions that overlap with these intervals will be
///                        excluded from normalization. If a region is in both "include_for_norm" and
///                        "exclude_for_norm", it will be excluded.
fn create_bedgraph_from_sorted_fragments<I, B>(
    fragments: I,
    chrom_sizes: &ReferenceSequences,
    bin_size: u64,
    smooth_base: Option<u64>,
    blacklist_regions: Option<&GIntervalMap<()>>,
    normalization: Option<Normalization>,
    include_for_norm: Option<&GIntervalMap<()>>,
    exclude_for_norm: Option<&GIntervalMap<()>>,
) -> Vec<BedGraph<f64>>
where
    I: Iterator<Item = B>,
    B: BEDLike,
{
    let mut norm_factor = 0u64;
    let mut bedgraph: Vec<BedGraph<f64>> = fragments.flat_map(|frag| {
        if blacklist_regions.map_or(false, |bl| bl.is_overlapped(&frag)) {
            None
        } else {
            if include_for_norm.map_or(true, |x| x.is_overlapped(&frag))
                && !exclude_for_norm.map_or(false, |x| x.is_overlapped(&frag))
            {
                norm_factor += frag.len();
            }
            let mut frag = BedGraph::from_bed(&frag, 1.0f64);
            fit_to_bin(&mut frag, bin_size);
            Some(frag)
        }
    })
    .merge_sorted_bedgraph()
        .flat_map(|x| clip_bed(x, chrom_sizes))
        .collect();

    let norm_factor = match normalization {
        None => 1.0,
        Some(Normalization::RPKM) => (norm_factor * bin_size) as f64 / 1e9,
        Some(Normalization::CPM) => norm_factor as f64 / 1e6,
        Some(Normalization::BPM) => {
            bedgraph
                .iter()
                .map(|x| x.value * (x.len() / bin_size) as f64)
                .sum::<f64>()
                / 1e6
        }
        Some(Normalization::RPGC) => todo!(),
    };

    bedgraph.iter_mut().for_each(|x| x.value /= norm_factor);

    if let Some(smooth_base) = smooth_base {
        let smooth_left = (smooth_base - 1) / 2;
        let smooth_right = smooth_base - 1 - smooth_left;
        bedgraph = smooth_bedgraph(bedgraph.into_iter(), smooth_left, smooth_right, chrom_sizes);
    }

    bedgraph
}

fn smooth_bedgraph<I>(
    input: I,
    left_window_len: u64,
    right_window_len: u64,
    chrom_sizes: &ReferenceSequences,
) -> Vec<BedGraph<f64>>
where
    I: Iterator<Item = BedGraph<f64>>,
{
    let mut key = 0;
    let mut prev = 0;
    input
        .chunk_by(|bed| {
            let k = if prev > bed.start().saturating_sub(left_window_len) {
                key
            } else {
                key += 1;
                key
            };
            prev = bed.end() + right_window_len;
            k
        })
        .into_iter()
        .flat_map(|(_, group)| smooth_bedgraph_block(group, left_window_len, right_window_len))
        .flat_map(|bed| clip_bed(bed, chrom_sizes))
        .collect()
}

/// Smooth the values in BedGraph. The input is expected to be overlapping blocks.
fn smooth_bedgraph_block<I>(
    data: I,
    ext_left: u64,
    ext_right: u64,
) -> impl Iterator<Item = BedGraph<f64>>
where
    I: IntoIterator<Item = BedGraph<f64>>,
{
    let n_bases = (ext_left + ext_right + 1) as f64;
    let mut data: Vec<_> = data
        .into_iter()
        .flat_map(|bed| {
            extend(bed.start(), bed.end(), ext_left, ext_right)
                .into_iter()
                .map(move |(s, e, n)| {
                    BedGraph::new(bed.chrom(), s, e, bed.value * n as f64 / n_bases)
                })
        })
        .collect();
    data.sort_unstable_by(|a, b| a.compare(b));
    data.into_iter().merge_sorted_bedgraph()
}

fn extend(start: u64, end: u64, ext_left: u64, ext_right: u64) -> Vec<(u64, u64, u64)> {
    let max = (end - start).min(ext_left + ext_right + 1);
    let s = start as i64 - ext_left as i64;
    let e = end as i64 + ext_right as i64;
    (s..e)
        .into_iter()
        .flat_map(move |i| {
            if i < 0 {
                None
            } else {
                let n = (i - s + 1).min(e - i).min(max as i64);
                Some((i as u64, n as u64))
            }
        })
        .chunk_by(|x| x.1)
        .into_iter()
        .map(|(k, group)| {
            let mut group = group.into_iter();
            let i = group.next().unwrap().0;
            let j = group.last().map_or(i, |x| x.0) + 1;
            (i, j, k)
        })
        .collect()
}

/// Create a bigwig file from BedGraph records.
fn create_bigwig_from_bedgraph<P, I>(
    bedgraph: I,
    chrom_sizes: &ReferenceSequences,
    filename: P,
) -> Result<()>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = BedGraph<f64>>,
{
    // write to bigwig file
    BigWigWrite::create_file(
        filename.as_ref().to_str().unwrap().to_string(),
        chrom_sizes
            .into_iter()
            .map(|(k, v)| (k.to_string(), usize::from(v.length()) as u32))
            .collect(),
    )?
    .write(
        bigtools::beddata::BedParserStreamingIterator::wrap_iter(
            bedgraph.into_iter().map(|x| {
                let val = bigtools::Value {
                    start: x.start() as u32,
                    end: x.end() as u32,
                    value: x.value as f32,
                };
                let res: Result<_, bigtools::bed::bedparser::BedValueError> =
                    Ok((x.chrom().to_string(), val));
                res
            }),
            false,
        ),
        tokio::runtime::Runtime::new().unwrap(),
    )?;
    Ok(())
}

fn clip_bed<B: BEDLike>(mut bed: B, chr_size: &ReferenceSequences) -> Option<B> {
    let size: usize = chr_size.get(&BString::from(bed.chrom()))?.length().into();
    let size = size as u64;
    if bed.start() >= size {
        return None;
    }
    if bed.end() > size {
        bed.set_end(size);
    }
    Some(bed)
}

fn fit_to_bin<B: BEDLike>(x: &mut B, bin_size: u64) {
    if bin_size > 1 {
        if x.start() % bin_size != 0 {
            x.set_start(x.start() - x.start() % bin_size);
        }
        if x.end() % bin_size != 0 {
            x.set_end(x.end() + bin_size - x.end() % bin_size);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    RPKM, // Reads per kilobase per million mapped reads. RPKM (per bin) =
    // number of reads per bin / (number of mapped reads (in millions) * bin length (kb)).
    CPM, // Counts per million mapped reads. CPM (per bin) =
    // number of reads per bin / number of mapped reads (in millions).
    BPM, // Bins Per Million mapped reads, same as TPM in RNA-seq. BPM (per bin) =
    // number of reads per bin / sum of all reads per bin (in millions).
    RPGC, // Reads per genomic content. RPGC (per bin) =
          // number of reads per bin / scaling factor for 1x average coverage.
}