use crate::genome::Connection;
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
};

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

const EXCESS_COEFFICIENT: f64 = 1.0;
const DISJOINT_COEFFICIENT: f64 = 1.0;
const WEIGHT_COEFFICIENT: f64 = 0.4;

fn delta(l: &Vec<Connection>, r: &Vec<Connection>) -> f64 {
    let l_size = l.len() as f64;
    let r_size = r.len() as f64;
    let fac = {
        let longest = f64::max(l_size, r_size);
        if longest < 20. {
            1.
        } else {
            longest
        }
    };

    if l_size == 0. || r_size == 0. {
        (EXCESS_COEFFICIENT * f64::max(l_size, r_size)) / fac
    } else {
        let (disjoint, excess) = disjoint_excess_count(l, r);
        (DISJOINT_COEFFICIENT * disjoint + EXCESS_COEFFICIENT * excess) / fac
            + WEIGHT_COEFFICIENT * avg_weight_diff(l, r)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::Connection;

    #[test]
    fn test_avg_weight_diff() {
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
    }

    #[test]
    fn test_avg_weight_diff_empty() {
        let full = vec![
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
                weight: -1.0,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 0,
                to: 0,
                weight: 2.0,
                enabled: true,
            },
        ];
        assert!((avg_weight_diff(&full, &vec![]) - 0.0).abs() < f64::EPSILON);
        assert!((avg_weight_diff(&vec![], &full,) - 0.0).abs() < f64::EPSILON);
        assert!((avg_weight_diff(&vec![], &vec![],) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_weight_diff_no_overlap() {
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
    }

    #[test]
    fn test_avg_weight_diff_no_diff() {
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
        assert_eq!(
            (4.0, 2.0),
            disjoint_excess_count(
                &vec![
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
                ],
                &vec![
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
                ]
            )
        );
    }

    #[test]
    fn test_disjoint_excess_count_symmetrical() {
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
        assert_eq!(disjoint_excess_count(&l, &r), disjoint_excess_count(&r, &l));
    }

    #[test]
    fn test_disjoint_excess_count_empty() {
        let full = vec![
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
        assert_eq!((0.0, 2.0), disjoint_excess_count(&full, &vec![]));
        assert_eq!((0.0, 2.0), disjoint_excess_count(&vec![], &full));
        assert_eq!((0.0, 0.0), disjoint_excess_count(&vec![], &vec![]));
    }

    #[test]
    fn test_disjoint_excess_count_no_overlap() {
        assert_eq!(
            (2.0, 2.0),
            disjoint_excess_count(
                &vec![
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
                ],
                &vec![
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
                ]
            )
        );
    }

    #[test]
    fn test_disjoint_excess_count_short_larger_inno() {
        assert_eq!(
            (3.0, 1.0),
            disjoint_excess_count(
                &vec![Connection {
                    inno: 10,
                    from: 0,
                    to: 0,
                    weight: 0.0,
                    enabled: true,
                }],
                &vec![
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
                ]
            )
        );
    }
}
