use crate::genome::Connection;
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
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

fn disjoint_excess_count(l: &Vec<Connection>, r: &Vec<Connection>) -> (f64, f64) {
    if l.is_empty() {
        (0., r.len() as f64)
    } else if r.is_empty() {
        (0., l.len() as f64)
    } else {
        let excess_boundary = min(l.last().unwrap().inno, r.last().unwrap().inno);

        let l_inno = l.iter().map(|c| c.inno).collect::<HashSet<_>>();
        let r_inno = r.iter().map(|c| c.inno).collect::<HashSet<_>>();
        l_inno
            .symmetric_difference(&r_inno)
            .fold((0., 0.), |(d, e), inno| {
                if *inno > excess_boundary {
                    (d, e + 1.)
                } else {
                    (d + 1., e)
                }
            })
    }
}

/// if genomes share no overlapping weights, their average diff should be 0
fn avg_weight_diff(l: &Vec<Connection>, r: &Vec<Connection>) -> f64 {
    let (short, long) = match (l.len(), r.len()) {
        (0, _) | (_, 0) => return 0.,
        (l_len, r_len) if l_len < r_len => (&l, &r),
        _ => (&r, &l),
    };

    let s_weights = short
        .iter()
        .map(|c| (c.inno, c.weight))
        .collect::<HashMap<usize, f64>>();

    let mut conut = 0.;
    let diff_sum = long
        .iter()
        .filter_map(
            |Connection {
                 inno, weight: l_w, ..
             }| {
                s_weights.get(inno).map(|s_w| {
                    conut += 1.;
                    (s_w - l_w).abs()
                })
            },
        )
        .sum::<f64>();

    if conut == 0. {
        0.
    } else {
        diff_sum / conut
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::{Connection, Genome};

    #[test]
    fn test_inno_gen() {
        let inno = inno_gen();
        assert_eq!(inno((0, 1)), 0);
        assert_eq!(inno((1, 2)), 1);
        assert_eq!(inno((0, 1)), 0);
    }

    #[test]
    fn test_avg_weight_diff() {
        // non-zero overlapping inno
        assert!(
            (avg_weight_diff(
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 1.0,
                        enabled: true
                    },
                ],
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.0,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -1.0,
                        enabled: true
                    },
                    Connection {
                        inno: 4,
                        from: 0,
                        to: 0,
                        weight: 2.0,
                        enabled: true
                    },
                ]
            ) - 0.5)
                .abs()
                < f64::EPSILON
        );

        // empty connections
        assert!(
            (avg_weight_diff(
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 1.0,
                        enabled: true
                    },
                ],
                &vec![]
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );

        // empty connections
        assert!(
            (avg_weight_diff(
                &vec![],
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.0,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -1.0,
                        enabled: true
                    },
                    Connection {
                        inno: 4,
                        from: 0,
                        to: 0,
                        weight: 2.0,
                        enabled: true
                    },
                ]
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );

        // zero overlapping inno
        assert!(
            (avg_weight_diff(
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 1.0,
                        enabled: true
                    },
                ],
                &vec![
                    Connection {
                        inno: 5,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 6,
                        from: 0,
                        to: 0,
                        weight: -0.5,
                        enabled: true
                    },
                ]
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );

        // varying weights
        assert!(
            (avg_weight_diff(
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.1,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: 0.2,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 0.3,
                        enabled: true
                    },
                ],
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.4,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 0.6,
                        enabled: true
                    },
                ]
            ) - 0.3)
                .abs()
                < f64::EPSILON
        );

        // weights with zero difference
        assert!(
            (avg_weight_diff(
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 1.0,
                        enabled: true
                    },
                ],
                &vec![
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 2,
                        from: 0,
                        to: 0,
                        weight: -0.5,
                        enabled: true
                    },
                    Connection {
                        inno: 3,
                        from: 0,
                        to: 0,
                        weight: 1.0,
                        enabled: true
                    },
                ]
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_disjoint_excess_count() {
        let l = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        let r = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 3,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true, // disjoint
            },
            Connection {
                inno: 4,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 5,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true, // excess
            },
        ];
        assert_eq!((1.0, 1.0), disjoint_excess_count(&l, &r));

        // two empty Vec<Connection>
        assert_eq!((0.0, 0.0), disjoint_excess_count(&vec![], &vec![]));

        // one empty Vec<Connection>
        let l = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true, // excess
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true, // excess
            },
        ];
        let r: Vec<Connection> = vec![];
        assert_eq!((0.0, 2.0), disjoint_excess_count(&l, &r));

        // one empty Vec<Connection> (reverse)
        let l: Vec<Connection> = vec![];
        let r = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        assert_eq!((0.0, 2.0), disjoint_excess_count(&l, &r)); // all genes in r are excess

        // no overlapping inno
        let l = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        let r = vec![
            Connection {
                inno: 3,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        assert_eq!((2.0, 2.0), disjoint_excess_count(&l, &r)); // inno 1 and 2 are disjoint, inno 3 and 4 are excess

        // both Vec<Connection> having their own disjoint genes
        let l = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 6,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        let r = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 3,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        assert_eq!((3.0, 1.0), disjoint_excess_count(&l, &r)); // inno 2, 3 and 4 are disjoint, inno 6 is excess

        // both Vec<Connection> having their own disjoint genes and r having one more gene with inno: 10
        let l = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 6,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        let r = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 3,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 8,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 10,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        assert_eq!((4.0, 2.0), disjoint_excess_count(&l, &r));

        // l having significantly fewer genes than r, but higher maximum inno
        let l = vec![Connection {
            inno: 10,
            from: 0,
            to: 0,
            weight: 0.0,
            enabled: true,
        }];
        let r = vec![
            Connection {
                inno: 1,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
            Connection {
                inno: 3,
                from: 0,
                to: 0,
                weight: 0.0,
                enabled: true,
            },
        ];
        assert_eq!((3.0, 1.0), disjoint_excess_count(&l, &r)); // inno 1, 2, and 3 are disjoint, inno 10 is excess
    }
}
