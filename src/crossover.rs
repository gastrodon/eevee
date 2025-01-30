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

/// if genomes share no overlapping weights, their average diff should be 0
fn avg_weight_diff(l: &Genome, r: &Genome) -> f64 {
    let (short, long) = match (l.connections.len(), r.connections.len()) {
        (0, _) | (_, 0) => return 0.,
        (l_len, r_len) if l_len < r_len => (&l, &r),
        _ => (&r, &l),
    };

    let s_weights = short
        .connections
        .iter()
        .map(|c| (c.inno, c.weight))
        .collect::<HashMap<usize, f64>>();

    let mut conut = 0.;
    let diff_sum = long
        .connections
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
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                },
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                    ],
                }
            ) - 0.5)
                .abs()
                < f64::EPSILON
        );

        // empty connections
        assert!(
            (avg_weight_diff(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                },
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![],
                }
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );

        // zero overlapping inno
        assert!(
            (avg_weight_diff(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                },
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                    ],
                }
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );

        // empty connections
        assert!(
            (avg_weight_diff(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![],
                },
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                    ],
                }
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );

        // varying weights
        assert!(
            (avg_weight_diff(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                },
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                    ],
                }
            ) - 0.3)
                .abs()
                < f64::EPSILON
        );

        // weights with zero difference
        assert!(
            (avg_weight_diff(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                },
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![
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
                }
            ) - 0.0)
                .abs()
                < f64::EPSILON
        );
    }
}
