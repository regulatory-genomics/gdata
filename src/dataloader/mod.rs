mod builder;
mod chunk;
mod index;
mod loader;

pub use builder::GenomeDataBuilder;
pub use loader::{CatGenomeDataLoader, GenomeDataLoader, GenomeDataLoaderMap};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::s;

    use super::*;
    use crate::w5z::W5Z;
    use rand::{
        distr::{Distribution, Uniform},
        Rng,
    };
    use std::io::Write;

    fn assert_almost_equal(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            let cutoff = x.abs() * tolerance;
            assert!(
                (x - y).abs() < cutoff,
                "Values {} and {} differ by more than {} percent",
                x,
                y,
                100.0 * tolerance
            );
        }
    }

    fn make_w5z(fl: PathBuf) -> PathBuf {
        let w5z = W5Z::new(fl.clone(), "w").unwrap();
        w5z.add(
            "chr1",
            &(1..9)
                .cycle()
                .take(8 * 128)
                .map(|x| x as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        w5z.add("chr2", &vec![0.0; 7919]).unwrap();
        fl
    }

    fn make_w5z2(fl: PathBuf) -> PathBuf {
        let mut rng = rand::rng();
        let uniform_dist = Uniform::new(-10000.0, 10000.0).unwrap();

        let w5z = W5Z::new(fl.clone(), "w").unwrap();
        w5z.add(
            "chr1",
            &(0..8 * 128)
                .map(|_| uniform_dist.sample(&mut rng))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        w5z.add(
            "chr2",
            &(0..7919)
                .map(|_| uniform_dist.sample(&mut rng))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fl
    }

    fn rand_dna(length: usize) -> String {
        let charset = "ACGT";
        let mut rng = rand::rng();
        let mut random_string = String::new();

        for _ in 0..length {
            let random_index = rng.random_range(0..charset.len());
            let random_char = charset.chars().nth(random_index).unwrap(); // .unwrap() is safe here as index is within bounds
            random_string.push(random_char);
        }
        random_string
    }

    fn make_fasta(fl: PathBuf) -> String {
        let chr1 = rand_dna(8 * 128);
        let chr2 = rand_dna(7919);
        let writer = std::fs::File::create(&fl).unwrap();
        let mut writer = std::io::BufWriter::new(writer);
        writeln!(writer, ">chr1").unwrap();
        writeln!(writer, "{}", &chr1).unwrap();

        writeln!(writer, ">chr2").unwrap();
        writeln!(writer, "{}", &chr2).unwrap();

        chr1 + &chr2
    }

    fn build_genome(path: PathBuf, window_size: u64, resolution: u64) -> (PathBuf, String) {
        let w5z1 = make_w5z(path.join("data1.w5z"));
        let w5z2 = make_w5z2(path.join("data2.w5z"));
        let fasta = make_fasta(path.join("genome.fa"));
        let builder = GenomeDataBuilder::new(
            path.join("builder"),
            path.join("genome.fa"),
            None,
            window_size,
            None,
            4,
            resolution,
            None,
            true,
        )
        .unwrap();
        builder.add_file("data1", w5z1).unwrap();
        builder.add_file("data2", w5z2).unwrap();
        (path.join("builder"), fasta)
    }

    #[test]
    fn test_genome_data_loader1() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (builder, fasta) = build_genome(temp_dir.path().to_path_buf(), 1, 1);

        for batch_size in 1..24 {
            let mut loader = GenomeDataLoader::new(
                builder.clone(),
                batch_size,
                None,
                None,
                None,
                None,
                false,
                false,
                1,
                0,
            )
            .unwrap();
            let (seq, values): (String, Vec<_>) = loader
                .iter()
                .flat_map(|(s, v)| {
                    s.into_strings()
                        .into_iter()
                        .zip(v.slice(s![.., .., 0]).to_owned().into_iter())
                })
                .collect();
            let w5z = W5Z::open(temp_dir.path().join("data1.w5z")).unwrap();
            let truth: Vec<_> = w5z
                .get("chr1")
                .unwrap()
                .into_iter()
                .chain(w5z.get("chr2").unwrap().into_iter())
                .collect();
            assert_eq!(values.len(), truth.len());
            assert_eq!(values, truth);
            assert_eq!(seq, fasta);

            let values: Vec<_> = loader
                .iter()
                .flat_map(|(_, v)| v.slice(s![.., .., 1]).to_owned().into_iter())
                .collect();
            let w5z = W5Z::open(temp_dir.path().join("data2.w5z")).unwrap();
            let truth: Vec<_> = w5z
                .get("chr1")
                .unwrap()
                .into_iter()
                .chain(w5z.get("chr2").unwrap().into_iter())
                .collect();
            assert_almost_equal(&values, &truth, 0.005);
        }
    }

    #[test]
    fn test_genome_data_loader2() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (builder, _) = build_genome(temp_dir.path().to_path_buf(), 128, 8);

        {
            let mut loader = GenomeDataLoader::new(
                builder.clone(),
                1,
                None,
                None,
                None,
                None,
                false,
                false,
                1,
                0,
            )
            .unwrap();
            for (i, (_, v)) in loader.iter().enumerate() {
                let v: Vec<_> = v.slice(s![.., .., 0]).into_iter().collect();
                assert!(v.len() == 128 / 8);
                if i < (8 * 128 / 128) {
                    assert!(v.iter().all(|&x| *x == 4.5));
                } else {
                    assert!(v.iter().all(|&x| *x == 0.0));
                }
            }
        }
        {
            let mut loader = GenomeDataLoader::new(
                builder.clone(),
                1,
                None,
                Some(2.0),
                Some(4.0),
                None,
                false,
                false,
                1,
                0,
            )
            .unwrap();
            for (i, (_, v)) in loader.iter().enumerate() {
                let v: Vec<_> = v.slice(s![.., .., 0]).into_iter().collect();
                assert!(v.len() == 128 / 8);
                if i < (8 * 128 / 128) {
                    v.iter().for_each(|&x| assert_eq!(*x, 4.0));
                } else {
                    assert!(v.iter().all(|&x| *x == 0.0));
                }
            }
        }
    }

    #[test]
    fn test_genome_data_loader3() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (builder, fasta) = build_genome(temp_dir.path().to_path_buf(), 32, 1);
        let w5z = W5Z::open(temp_dir.path().join("data2.w5z")).unwrap();
        let truth: Vec<_> = w5z
            .get("chr1")
            .unwrap()
            .into_iter()
            .chain(w5z.get("chr2").unwrap().into_iter())
            .collect();

        let mut loader = GenomeDataLoader::new(
            builder.clone(),
            7,
            None,
            None,
            None,
            None,
            false,
            false,
            1,
            0,
        )
        .unwrap();
        let mut values: Vec<_> = loader
            .iter()
            .flat_map(|(_, v)| v.slice(s![.., .., 1]).to_owned().into_iter())
            .collect();
        values = values[0..truth.len()].to_vec();
        assert_almost_equal(&values, &truth, 0.005);

        let mut loader = GenomeDataLoader::new(
            builder.clone(),
            7,
            None,
            None,
            None,
            Some(16),
            false,
            false,
            1,
            0,
        )
        .unwrap();
        let mut seqs = String::new();
        let mut values = Vec::new();
        loader
            .iter()
            .for_each(|(s, v)| {
                assert!(
                    v.shape()[1] == 16,
                    "Expected 16 channels, got {}",
                    v.shape()[1]
                );
                seqs.extend(s.into_strings());
                values.extend(v.slice(s![.., .., 1]).to_owned().into_iter());
            });
        values = values[0..truth.len()].to_vec();
        assert_eq!(values.len(), truth.len());
        assert_almost_equal(&values, &truth, 0.005);
        assert_eq!(seqs[0..fasta.len()], fasta);
    }

    #[test]
    fn test_genome_data_loader4() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (builder, _) = build_genome(temp_dir.path().to_path_buf(), 32, 1);

        let mut loader1 = GenomeDataLoader::new(
            builder.clone(),
            7,
            Some(4),
            None,
            None,
            None,
            false,
            false,
            1,
            0,
        )
        .unwrap();
        let values1: Vec<_> = loader1
            .iter()
            .flat_map(|(_, v)| v.slice(s![.., .., 1]).to_owned().into_iter())
            .collect();

        let mut loader2 = GenomeDataLoader::new(
            builder.clone(),
            7,
            None,
            None,
            None,
            None,
            false,
            false,
            1,
            0,
        )
        .unwrap();
        let values2: Vec<_> = loader2
            .iter()
            .flat_map(|(_, v)| {
                let arr = v.slice(s![.., 4..28, 1]).to_owned();
                arr.into_iter()
            })
            .collect();
        assert_eq!(values1.len(), values2.len());
        assert_eq!(values1, values2);
    }

    #[test]
    fn test_genome_data_loader5() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (builder, fasta) = build_genome(temp_dir.path().to_path_buf(), 32, 1);

        let mut loader1 = GenomeDataLoader::new(
            builder.clone(),
            7,
            Some(4),
            None,
            None,
            Some(16),
            false,
            false,
            1,
            0,
        )
        .unwrap();
        let mut seqs = String::new();
        let mut values1 = Vec::new();
        loader1
            .iter()
            .for_each(|(s, v)| {
                seqs.extend(s.into_strings());
                values1.extend(v.slice(s![.., .., 1]).to_owned().into_iter());
            });

        let mut loader2 = GenomeDataLoader::new(
            builder.clone(),
            7,
            None,
            None,
            None,
            Some(16),
            false,
            false,
            1,
            0,
        )
        .unwrap();
        let values2: Vec<_> = loader2
            .iter()
            .flat_map(|(_, v)| {
                let arr = v.slice(s![.., 4..12, 1]).to_owned();
                arr.into_iter()
            })
            .collect();
        assert_eq!(values1.len(), values2.len());
        assert_eq!(values1, values2);
        assert_eq!(seqs[0..fasta.len()], fasta);
    }
}