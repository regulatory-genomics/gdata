mod builder;
mod chunk;
mod index;
mod loader;

pub use builder::GenomeDataBuilder;
pub use loader::{GenomeDataLoader, GenomeDataLoaderMap};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::w5z::W5Z;
    use std::io::Write;

    fn make_w5z(fl: PathBuf) -> PathBuf {
        let w5z = W5Z::new(fl.clone(), "w").unwrap();
        w5z.add("chr1", &vec![0.0; 7919]).unwrap();
        w5z.add(
            "chr2",
            &(1..9)
                .cycle()
                .take(8 * 128)
                .map(|x| x as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fl
    }

    fn make_fasta(fl: PathBuf) -> PathBuf {
        let writer = std::fs::File::create(&fl).unwrap();
        let mut writer = std::io::BufWriter::new(writer);
        writeln!(writer, ">chr1").unwrap();
        writeln!(
            writer,
            "{}",
            &std::iter::repeat_n('A', 7919).collect::<String>()
        )
        .unwrap();
        writeln!(writer, ">chr2").unwrap();
        writeln!(
            writer,
            "{}",
            &std::iter::repeat_n('C', 8 * 128).collect::<String>()
        )
        .unwrap();
        fl
    }

    fn build_genome(path: PathBuf) -> PathBuf {
        let window_size = 128;
        let resolution = 8;

        let w5z = make_w5z(path.join("data.w5z"));
        let fasta = make_fasta(path.join("genome.fa"));
        let builder = GenomeDataBuilder::new(
            path.join("builder"),
            fasta,
            None,
            window_size,
            None,
            4,
            resolution,
            None,
            true,
        )
        .unwrap();
        builder.add_file("data", w5z).unwrap();
        path.join("builder")
    }

    #[test]
    fn test_genome_data_builder() {
        let temp_dir = tempfile::tempdir().unwrap();
        build_genome(temp_dir.path().to_path_buf());
    }

    #[test]
    fn test_genome_data_loader() {
        let temp_dir = tempfile::tempdir().unwrap();
        let builder = build_genome(temp_dir.path().to_path_buf());

        {
            let loader =
                GenomeDataLoader::new(builder.clone(), 1, None, None, None, false, 1).unwrap();
            for (i, (_, v)) in loader.iter().enumerate() {
                let (v, _) = v.into_raw_vec_and_offset();
                assert!(v.len() == 128 / 8);
                if i < (7919 / 128 + 1) {
                    assert!(v.iter().all(|&x| x == 0.0));
                } else {
                    assert!(v.iter().all(|&x| x == 4.5));
                }
            }
        }
        {
            let loader =
                GenomeDataLoader::new(builder.clone(), 1, None, Some(2.0), Some(4.0), false, 1)
                    .unwrap();
            for (i, (_, v)) in loader.iter().enumerate() {
                let (v, _) = v.into_raw_vec_and_offset();
                assert!(v.len() == 128 / 8);
                if i < (7919 / 128 + 1) {
                    assert!(v.iter().all(|&x| x == 0.0));
                } else {
                    v.iter().for_each(|&x| assert_eq!(x, 4.0));
                }
            }
        }
    }
}