use crate::genome::Connection;
use rand::{rngs::ThreadRng, Rng};
use std::{
    cmp::{min, Ordering},
    collections::{HashMap, HashSet},
};

fn disjoint_excess_count(l: &[Connection], r: &[Connection]) -> (f64, f64) {
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
fn avg_weight_diff(l: &[Connection], r: &[Connection]) -> f64 {
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

pub fn delta(l: &[Connection], r: &[Connection]) -> f64 {
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

/// crossover connections where l and r are equally fit
fn crossover_eq(
    l: &HashMap<usize, &Connection>,
    r: &HashMap<usize, &Connection>,
    rng: &mut ThreadRng,
) -> Vec<Connection> {
    let keys: HashSet<_> = HashSet::from_iter(l.keys().chain(r.keys()).cloned());

    keys.iter()
        .map(|inno| {
            // TODO 75% chance to disable gene if disabled in either parent
            (*match (l.get(inno), r.get(inno)) {
                (None, None) => unreachable!(),
                (None, Some(conn)) | (Some(conn), None) => conn,
                (Some(l_conn), Some(r_conn)) => {
                    if rng.random_bool(0.5) {
                        l_conn
                    } else {
                        r_conn
                    }
                }
            })
            .clone()
        })
        .collect()
}

/// crossover connections where l is more fit than r
fn crossover_ne(
    l: &HashMap<usize, &Connection>,
    r: &HashMap<usize, &Connection>,
    rng: &mut ThreadRng,
) -> Vec<Connection> {
    l.iter()
        .map(|(inno, l_conn)| {
            // TODO 75% chance to disable gene if disabled in either parent
            (*if r.contains_key(inno) && rng.random_bool(0.5) {
                r.get(inno).unwrap()
            } else {
                l_conn
            })
            .clone()
        })
        .collect()
}

/// crossover connections
/// l_fit describes how fit l is compared to r,
pub fn crossover(
    l: &[Connection],
    r: &[Connection],
    l_fit: Ordering,
    rng: &mut ThreadRng,
) -> Vec<Connection> {
    let lookup_l = l.iter().map(|conn| (conn.inno, conn)).collect();
    let lookup_r = r.iter().map(|conn| (conn.inno, conn)).collect();

    let mut usort = match l_fit {
        Ordering::Equal => crossover_eq(&lookup_l, &lookup_r, rng),
        Ordering::Less => crossover_ne(&lookup_r, &lookup_l, rng),
        Ordering::Greater => crossover_ne(&lookup_l, &lookup_r, rng),
    };

    usort.sort_by_key(|c| c.inno);
    usort
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::Connection;
    use rand::rng;

    #[test]
    fn test_avg_weight_diff() {
        assert!(
            (avg_weight_diff(
                &[
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
                    }
                ],
                &[
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
                    }
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
        assert!((avg_weight_diff(&full, &[]) - 0.0).abs() < f64::EPSILON);
        assert!((avg_weight_diff(&[], &full,) - 0.0).abs() < f64::EPSILON);
        assert!((avg_weight_diff(&[], &[],) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_weight_diff_no_overlap() {
        assert!(
            (avg_weight_diff(
                &[
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
                &[
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
                &[
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
                &[
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
                &[
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
                &[
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
        assert_eq!((0.0, 2.0), disjoint_excess_count(&full, &[]));
        assert_eq!((0.0, 2.0), disjoint_excess_count(&[], &full));
        assert_eq!((0.0, 0.0), disjoint_excess_count(&[], &[]));
    }

    #[test]
    fn test_disjoint_excess_count_no_overlap() {
        assert_eq!(
            (2.0, 2.0),
            disjoint_excess_count(
                &[
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
                &[
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
                &[Connection {
                    inno: 10,
                    from: 0,
                    to: 0,
                    weight: 0.0,
                    enabled: true,
                }],
                &[
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

    #[test]
    fn test_crossover_eq() {
        let l = [
            Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 0.6,
                enabled: true,
            },
            Connection {
                inno: 1,
                from: 1,
                to: 2,
                weight: 1.,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 2,
                to: 1,
                weight: 1.2,
                enabled: true,
            },
        ];
        let r = [
            Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 0.3,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 2,
                to: 1,
                weight: 0.2,
                enabled: false,
            },
            Connection {
                inno: 3,
                from: 2,
                to: 3,
                weight: 1.,
                enabled: true,
            },
        ];

        for _ in 0..1000 {
            let lr = crossover(&l, &r, Ordering::Equal, &mut rng());

            assert_eq!(lr.len(), 4);
            assert!(lr[0] == l[0] || lr[0] == r[0]);
            assert_eq!(lr[1], l[1]);
            assert!(lr[2] == l[2] || lr[2] == r[1]);
            assert_eq!(lr[3], r[2])
        }
    }

    #[test]
    fn test_crossover_gt() {
        let l = [
            Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 0.6,
                enabled: true,
            },
            Connection {
                inno: 1,
                from: 1,
                to: 2,
                weight: 1.,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 2,
                to: 1,
                weight: 1.2,
                enabled: true,
            },
        ];
        let r = [
            Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 0.3,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 2,
                to: 1,
                weight: 0.2,
                enabled: false,
            },
            Connection {
                inno: 3,
                from: 2,
                to: 3,
                weight: 1.,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 2,
                to: 4,
                weight: 1.,
                enabled: true,
            },
        ];

        for _ in 0..1000 {
            let lr = crossover(&l, &r, Ordering::Greater, &mut rng());

            assert_eq!(lr.len(), l.len());
            assert!(lr[0] == l[0] || lr[0] == r[0]);
            assert_eq!(lr[1], l[1]);
            assert!(lr[2] == l[2] || lr[2] == r[1]);
        }
    }

    #[test]
    fn test_crossover_lt() {
        let l = [
            Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 0.6,
                enabled: true,
            },
            Connection {
                inno: 1,
                from: 1,
                to: 2,
                weight: 1.,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 2,
                to: 1,
                weight: 1.2,
                enabled: true,
            },
        ];
        let r = [
            Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 0.3,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 2,
                to: 1,
                weight: 0.2,
                enabled: false,
            },
            Connection {
                inno: 3,
                from: 2,
                to: 3,
                weight: 1.,
                enabled: true,
            },
            Connection {
                inno: 4,
                from: 2,
                to: 4,
                weight: 1.,
                enabled: true,
            },
        ];

        for _ in 0..1000 {
            let lr = crossover(&l, &r, Ordering::Less, &mut rng());

            assert_eq!(lr.len(), r.len());
            assert!(lr[0] == l[0] || lr[0] == r[0]);
            assert!(lr[1] == l[2] || lr[1] == r[1]);
            assert_eq!(lr[2], r[2]);
            assert_eq!(lr[3], r[3]);
        }
    }
}
