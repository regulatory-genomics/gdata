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
    use rand::distr::{Distribution, Uniform};
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

    fn make_fasta(fl: PathBuf) -> PathBuf {
        let writer = std::fs::File::create(&fl).unwrap();
        let mut writer = std::io::BufWriter::new(writer);
        writeln!(writer, ">chr1").unwrap();
        writeln!(
            writer,
            "{}",
            &std::iter::repeat_n('C', 8 * 128).collect::<String>()
        )
        .unwrap();

        writeln!(writer, ">chr2").unwrap();
        writeln!(
            writer,
            "{}",
            &std::iter::repeat_n('A', 7919).collect::<String>()
        )
        .unwrap();
        fl
    }

    fn build_genome(path: PathBuf, window_size: u64, resolution: u64) -> PathBuf {
        let w5z1 = make_w5z(path.join("data1.w5z"));
        let w5z2 = make_w5z2(path.join("data2.w5z"));
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
        builder.add_file("data1", w5z1).unwrap();
        builder.add_file("data2", w5z2).unwrap();
        path.join("builder")
    }

    #[test]
    fn test_genome_data_builder() {
        let temp_dir = tempfile::tempdir().unwrap();
        build_genome(temp_dir.path().to_path_buf(), 128, 8);
    }

    #[test]
    fn test_genome_data_loader1() {
        let temp_dir = tempfile::tempdir().unwrap();
        let builder = build_genome(temp_dir.path().to_path_buf(), 1, 1);

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
            let values: Vec<_> = loader
                .iter()
                .flat_map(|(_, v)| v.slice(s![.., .., 0]).to_owned().into_iter())
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
        let builder = build_genome(temp_dir.path().to_path_buf(), 128, 8);

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
        let builder = build_genome(temp_dir.path().to_path_buf(), 32, 1);
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
            1,
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
        let mut values: Vec<_> = loader
            .iter()
            .flat_map(|(_, v)| {
                assert!(
                    v.shape()[1] == 16,
                    "Expected 16 channels, got {}",
                    v.shape()[1]
                );
                v.slice(s![.., .., 1]).to_owned().into_iter()
            })
            .collect();
        values = values[0..truth.len()].to_vec();
        assert_eq!(values.len(), truth.len());
        assert_almost_equal(&values, &truth, 0.005);
    }
}
