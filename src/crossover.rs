use crate::genome::Connection;
use core::cmp::Ordering;
use rand::{rngs::ThreadRng, Rng};
use std::collections::{HashMap, HashSet};

pub fn disjoint_excess_count(l: &[Connection], r: &[Connection]) -> (f64, f64) {
    let mut l_iter = l.iter();
    let mut r_iter = r.iter();

    let mut l_conn = match l_iter.next() {
        Some(c) => c,
        None => return (0., r_iter.count() as f64),
    };

    let mut r_conn = match r_iter.next() {
        Some(c) => c,
        None => return (0., l_iter.count() as f64 + 1.),
    };

    let mut disjoint = 0.;
    let excess_passed = loop {
        match l_conn.inno.cmp(&r_conn.inno) {
            Ordering::Equal => {
                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break 0.,
                };

                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break 1.,
                };
            }
            Ordering::Greater => {
                disjoint += 1.;
                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break 1.,
                }
            }
            Ordering::Less => {
                disjoint += 1.;
                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break 1.,
                }
            }
        }
    };

    (
        disjoint,
        l_iter.count() as f64 + r_iter.count() as f64 + excess_passed,
    )
}

/// if genomes share no overlapping weights, their average diff should be 0
pub fn avg_weight_diff(l: &[Connection], r: &[Connection]) -> f64 {
    let mut diff = 0.;
    let mut count = 0.;
    let mut l_iter = l.iter();
    let mut r_iter = r.iter();

    let mut l_conn = match l_iter.next() {
        Some(c) => c,
        None => return 0.,
    };

    let mut r_conn = match r_iter.next() {
        Some(c) => c,
        None => return 0.,
    };

    loop {
        match l_conn.inno.cmp(&r_conn.inno) {
            Ordering::Equal => {
                diff += (l_conn.weight - r_conn.weight).abs();
                count += 1.;

                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break,
                };

                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break,
                };
            }
            Ordering::Greater => {
                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break,
                }
            }
            Ordering::Less => {
                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break,
                }
            }
        }
    }

    if count == 0. {
        0.
    } else {
        diff / count
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

const CHANCE_PICK_LR: f64 = 0.5;
const CHANCE_KEEP_DISABLED: f64 = 0.75;
const CHANCE_RAND_DISABLED: f64 = 0.01;

#[inline]
fn pick_gene(
    base_conn: &Connection,
    opt_conn: Option<&&Connection>,
    rng: &mut ThreadRng,
) -> Connection {
    let mut conn = if let Some(r_conn) = opt_conn {
        (*if rng.random_bool(CHANCE_PICK_LR) {
            r_conn
        } else {
            base_conn
        })
        .to_owned()
    } else {
        base_conn.to_owned()
    };

    conn.enabled = if ((!base_conn.enabled || opt_conn.is_some_and(|r_conn| !r_conn.enabled))
        && rng.random_bool(CHANCE_KEEP_DISABLED))
        || rng.random_bool(CHANCE_RAND_DISABLED)
    {
        false
    } else {
        conn.enabled
    };

    conn
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
            match (l.get(inno), r.get(inno)) {
                (None, None) => unreachable!(),
                (opt_conn, Some(base_conn)) | (Some(base_conn), opt_conn) => {
                    pick_gene(base_conn, opt_conn, rng)
                }
            }
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
        .map(|(inno, l_conn)| pick_gene(l_conn, r.get(inno), rng))
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
        let diff = avg_weight_diff(
            &[
                Connection {
                    inno: 1,
                    from: 0,
                    to: 0,
                    weight: 0.5,
                    enabled: true,
                },
                Connection {
                    inno: 2,
                    from: 0,
                    to: 0,
                    weight: -0.5,
                    enabled: true,
                },
                Connection {
                    inno: 3,
                    from: 0,
                    to: 0,
                    weight: 1.0,
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
            ],
        );
        assert!((diff - 0.5).abs() < f64::EPSILON, "diff ne: {diff}, 0.5");
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

        let diff = avg_weight_diff(&full, &[]);
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");

        let diff = avg_weight_diff(&[], &full);
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");

        let diff = avg_weight_diff(&[], &[]);
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");
    }

    #[test]
    fn test_avg_weight_diff_no_overlap() {
        let diff = avg_weight_diff(
            &[
                Connection {
                    inno: 1,
                    from: 0,
                    to: 0,
                    weight: 0.5,
                    enabled: true,
                },
                Connection {
                    inno: 2,
                    from: 0,
                    to: 0,
                    weight: -0.5,
                    enabled: true,
                },
                Connection {
                    inno: 3,
                    from: 0,
                    to: 0,
                    weight: 1.0,
                    enabled: true,
                },
            ],
            &[
                Connection {
                    inno: 5,
                    from: 0,
                    to: 0,
                    weight: 0.5,
                    enabled: true,
                },
                Connection {
                    inno: 6,
                    from: 0,
                    to: 0,
                    weight: -0.5,
                    enabled: true,
                },
            ],
        );
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");
    }

    #[test]
    fn test_avg_weight_diff_no_diff() {
        let diff = avg_weight_diff(
            &[
                Connection {
                    inno: 1,
                    from: 0,
                    to: 0,
                    weight: 0.5,
                    enabled: true,
                },
                Connection {
                    inno: 2,
                    from: 0,
                    to: 0,
                    weight: -0.5,
                    enabled: true,
                },
                Connection {
                    inno: 3,
                    from: 0,
                    to: 0,
                    weight: 1.0,
                    enabled: true,
                },
            ],
            &[
                Connection {
                    inno: 1,
                    from: 0,
                    to: 0,
                    weight: 0.5,
                    enabled: true,
                },
                Connection {
                    inno: 2,
                    from: 0,
                    to: 0,
                    weight: -0.5,
                    enabled: true,
                },
                Connection {
                    inno: 3,
                    from: 0,
                    to: 0,
                    weight: 1.0,
                    enabled: true,
                },
            ],
        );
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");
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
    fn test_disjoint_excess_count_hanging_l() {
        assert_eq!(
            (0.0, 1.0),
            disjoint_excess_count(
                &[
                    Connection {
                        inno: 0,
                        from: 0,
                        to: 0,
                        weight: 0.0,
                        enabled: true,
                    },
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
                        inno: 0,
                        from: 0,
                        to: 0,
                        weight: 0.0,
                        enabled: true,
                    },
                    Connection {
                        inno: 1,
                        from: 0,
                        to: 0,
                        weight: 0.0,
                        enabled: true,
                    },
                ]
            )
        )
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
            {
                let mut lr_2 = lr[2].to_owned();
                lr_2.enabled = false;
                let mut l_2 = l[2].to_owned();
                l_2.enabled = false;
                let mut r_1 = r[1].to_owned();
                r_1.enabled = false;
                assert!(lr_2 == l_2 || lr_2 == r_1)
            };
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
            {
                let mut lr_2 = lr[2].to_owned();
                lr_2.enabled = false;
                let mut l_2 = l[2].to_owned();
                l_2.enabled = false;
                let mut r_1 = r[1].to_owned();
                r_1.enabled = false;
                assert!(
                    lr_2 == l_2 || lr_2 == r_1,
                    "base:{lr_2:?}\nl   :{l_2:?}\nr   :{r_1:?}"
                );
            }
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
            {
                let mut lr_1 = lr[1].to_owned();
                lr_1.enabled = false;
                let mut l_2 = l[2].to_owned();
                l_2.enabled = false;
                let mut r_1 = r[1].to_owned();
                r_1.enabled = false;
                assert!(lr_1 == l_2 || lr_1 == r_1);
            }
            assert_eq!(lr[2], r[2]);
            assert_eq!(lr[3], r[3]);
        }
    }
}
