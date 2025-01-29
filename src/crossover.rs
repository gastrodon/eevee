use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

fn inno_gen() -> impl Fn((usize, usize)) -> usize {
    let head = Arc::new(Mutex::new(0));
    let inno = Arc::new(Mutex::new(HashMap::<(usize, usize), usize>::new()));
    return move |v: (usize, usize)| {
        let mut head = head.lock().unwrap();
        let mut inno = inno.lock().unwrap();
        match inno.get(&v) {
            Some(n) => *n,
            None => {
                let n = *head;
                *head += 1;
                inno.insert(v, n);
                n
            }
        }
    };
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inno_gen() {
        let inno = inno_gen();
        assert_eq!(inno((0, 1)), 0);
        assert_eq!(inno((1, 2)), 1);
        assert_eq!(inno((0, 1)), 0);
    }
}
