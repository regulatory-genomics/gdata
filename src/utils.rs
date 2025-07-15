use std::thread;
use crossbeam::channel::{bounded, Receiver};

pub struct PrefetchIterator<T> {
    receiver: Receiver<T>,
}

impl<T: Send + 'static> PrefetchIterator<T> {
    pub fn new<I>(iter: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        let (sender, receiver) = bounded(buffer_size);

        thread::spawn(move || {
            for item in iter {
                if sender.send(item).is_err() {
                    break;
                }
            }
            // sender dropped, receiver will return None
        });

        PrefetchIterator { receiver }
    }
}

impl<T> Iterator for PrefetchIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_iterator() {
        let data = 0..100;
        let prefetch_iter = PrefetchIterator::new(data, 10); // prefetch buffer size 10

        for item in prefetch_iter {
            println!("Got: {}", item);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}