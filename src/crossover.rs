//! Functions related to performing measuring compatability for and performing crossover
//! reproduction.

use crate::genome::Connection;
use core::cmp::Ordering;
use rand::RngCore;

/// Count misaligned [Connection]s between 2 slices. Where `l` is more fit ( TODO really? ), we
/// consider disjoint genes to be misalignments of innovation ids < `r`s max, and excess are
/// misalignments of ids > `r`s max.
fn disjoint_excess_count<C: Connection>(l: &[C], r: &[C]) -> (f64, f64) {
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
        match l_conn.inno().cmp(&r_conn.inno()) {
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

/// Average param difference between aligned genes from `l` and `r`. Misaligned genes are not
/// considered
fn avg_param_diff<C: Connection>(l: &[C], r: &[C]) -> f64 {
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
        match l_conn.inno().cmp(&r_conn.inno()) {
            Ordering::Equal => {
                diff += l_conn.param_diff(r_conn);
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

/// difference between [Connection]s in terms of crossover compatability. Higher deltas tend to
/// yield more destructive crossover.
pub fn delta<C: Connection>(l: &[C], r: &[C]) -> f64 {
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
        (C::EXCESS_COEFFICIENT * f64::max(l_size, r_size)) / fac
    } else {
        let (disjoint, excess) = disjoint_excess_count(l, r);
        (C::DISJOINT_COEFFICIENT * disjoint + C::EXCESS_COEFFICIENT * excess) / fac
            + C::PARAM_COEFFICIENT * avg_param_diff(l, r)
    }
}

#[inline]
fn pick_gene<C: Connection>(base_conn: &C, opt_conn: Option<&C>, rng: &mut impl RngCore) -> C {
    let mut conn = if let Some(r_conn) = opt_conn {
        // TODO be able to differentiate PickLEQ and PickLNE
        if rng.next_u64() < C::PROBABILITY_PICK_RL {
            r_conn
        } else {
            base_conn
        }
        .to_owned()
    } else {
        base_conn.to_owned()
    };

    // TODO It seems like it will always check RAND_DISABLED, and sometimes
    // check KEEP_DISABLED. I wonder if checking RAND_DISABLED first would bypass
    // RAND_DISABLED% of checks that would then check KEEP_DISABLED?
    if (!base_conn.enabled() || opt_conn.is_some_and(|r_conn| !r_conn.enabled()))
        && rng.next_u64() < C::PROBABILITY_KEEP_DISABLED
    {
        conn.disable();
    }

    conn
}

/// crossover connections where l and r are equally fit
fn crossover_eq<C: Connection>(l: &[C], r: &[C], rng: &mut impl RngCore) -> Vec<C> {
    // TODO I wonder what the actual average case overlap between genomes is?
    // probably pretty close, could we measure this?
    let mut cross = Vec::with_capacity(l.len() + r.len());
    let mut l_idx = 0;
    let mut r_idx = 0;
    loop {
        match (l.get(l_idx), r.get(r_idx)) {
            (None, None) => break,
            (None, Some(_)) => {
                // TODO is it faster to extend, or to loop-push?
                cross.extend(r[r_idx..].iter().map(|conn| pick_gene(conn, None, rng)));
                break;
            }
            (Some(_), None) => {
                cross.extend(l[l_idx..].iter().map(|conn| pick_gene(conn, None, rng)));
                break;
            }
            (Some(l_conn), Some(r_conn)) => match l_conn.inno().cmp(&r_conn.inno()) {
                Ordering::Equal => {
                    cross.push(pick_gene(l_conn, Some(r_conn), rng));
                    l_idx += 1;
                    r_idx += 1;
                }
                Ordering::Less => {
                    cross.push(pick_gene(l_conn, None, rng));
                    l_idx += 1;
                }
                Ordering::Greater => {
                    cross.push(pick_gene(r_conn, None, rng));
                    r_idx += 1;
                }
            },
        }
    }

    cross.shrink_to_fit(); // TODO what happens if I remove this
    cross
}

/// crossover connections where l is more fit than r
fn crossover_ne<C: Connection>(l: &[C], r: &[C], rng: &mut impl RngCore) -> Vec<C> {
    // copy l, pick_gene where l.inno() == r.inno()
    let mut cross = Vec::with_capacity(l.len());
    let mut r_idx = 0;
    for l_conn in l {
        // TODO is r_idx < r.len() && r[r_idx] or maybe even get_unchecked
        while r
            .get(r_idx)
            .is_some_and(|r_conn| r_conn.inno() < l_conn.inno())
        {
            r_idx += 1;
        }

        // TODO above applies here
        cross.push(pick_gene(
            l_conn,
            r.get(r_idx)
                .is_some_and(|r_conn| r_conn.inno() == l_conn.inno())
                .then(|| &r[r_idx]),
            rng,
        ))
    }

    cross
}

/// Perform crossover reproduction across 2 [Connection] slices `l` and `r`. `l_fit` describes
/// how fit `l` is compared to `r`, which determines who's genes to prioritize when misaligned.
pub fn crossover<C: Connection>(
    l: &[C],
    r: &[C],
    l_fit: Ordering,
    rng: &mut impl RngCore,
) -> Vec<C> {
    let mut usort = match l_fit {
        Ordering::Equal => crossover_eq(l, r, rng),
        Ordering::Less => crossover_ne(r, l, rng),
        Ordering::Greater => crossover_ne(l, r, rng),
    };

    usort.sort_by_key(|c| c.inno());
    usort
}

#[cfg(test)]
mod bench {
    use crate::{
        crossover::{avg_param_diff, crossover, disjoint_excess_count},
        genome::WConnection,
        random::default_rng,
        test_data,
    };
    use core::cmp::Ordering;
    use criterion::Criterion;
    use criterion_macro::criterion;

    #[criterion]
    fn bench_distance(bench: &mut Criterion) {
        type C = WConnection;

        let l_conn =
            serde_json::from_str::<Vec<C>>(include_str!("../test-data/ctr-connection-rand-l.json"))
                .unwrap();
        let r_conn =
            serde_json::from_str::<Vec<C>>(include_str!("../test-data/ctr-connection-rand-r.json"))
                .unwrap();

        bench.bench_function("disjoint-excess-count", |b| {
            b.iter(|| disjoint_excess_count(&l_conn, &r_conn))
        });

        bench.bench_function("avg-weight-diff", |b| {
            b.iter(|| avg_param_diff(&l_conn, &r_conn))
        });
    }

    #[criterion]
    fn bench_crossover(bench: &mut Criterion) {
        type C = WConnection;

        let l_conn =
            serde_json::from_str::<Vec<C>>(test_data!("ctr-connection-rand-l.json")).unwrap();
        let r_conn =
            serde_json::from_str::<Vec<C>>(test_data!("ctr-connection-rand-r.json")).unwrap();

        let mut rng = default_rng();
        bench.bench_function("crossover-ne", |b| {
            b.iter(|| crossover(&l_conn, &r_conn, Ordering::Greater, &mut rng))
        });

        bench.bench_function("crossover-eq", |b| {
            b.iter(|| crossover(&l_conn, &r_conn, Ordering::Equal, &mut rng))
        });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        assert_f64_approx, assert_some_normalized,
        genome::{connection::BWConnection, WConnection},
        new_t,
        random::default_rng,
        test_t,
    };
    use std::collections::{HashMap, HashSet};

    test_t!(
    test_avg_param_diff[T: WConnection]() {
        let diff = avg_param_diff(
            &[
                new_t!(inno = 1, weight = 0.5,),
                new_t!(inno = 2, weight = -0.5,),
                new_t!(inno = 3, weight = 1.0,),
            ],
            &[
                new_t!(inno = 1, weight = 0.0,),
                new_t!(inno = 2, weight = -1.0,),
                new_t!(inno = 4, weight = 2.0,),
            ],
        );
        assert_f64_approx!(diff, 0.5, "diff ne: {diff}, 0.5");
    });

    test_t!(
    test_avg_param_diff[T: BWConnection]() {
        let diff = avg_param_diff(
            &[
                new_t!(inno = 1, weight = 0.5, bias = 1.),
                new_t!(inno = 2, weight = -0.5,),
                new_t!(inno = 3, weight = 1.0,),
            ],
            &[
                new_t!(inno = 1, weight = 0.0, bias = 0.),
                new_t!(inno = 2, weight = -1.0,),
                new_t!(inno = 4, weight = 2.0,),
            ],
        );
        let diff_w = 0.5;
        let diff_b = 1. / 2.;
        assert_f64_approx!(diff, diff_w + diff_b, "diff ne: {diff}, 0.5");
    });

    test_t!(
    test_avg_param_diff_empty[T: WConnection | BWConnection]() {
        let full = vec![
            new_t!(inno = 1, weight = 0.0,),
            new_t!(inno = 2, weight = -1.0,),
            new_t!(inno = 4, weight = 2.0,),
        ];

        let diff = avg_param_diff(&full, &[]);
        assert_f64_approx!(diff, 0.0, "diff ne: {diff}, 0.");

        let diff = avg_param_diff(&[], &full);
        assert_f64_approx!(diff, 0.0, "diff ne: {diff}, 0.");

        let diff = avg_param_diff::<T>(&[], &[]);
        assert_f64_approx!(diff, 0.0, "diff ne: {diff}, 0.");
    });

    test_t!(
    test_avg_param_diff_no_overlap[T: WConnection | BWConnection]() {
        let diff = avg_param_diff(
            &[
                new_t!(inno = 1, weight = 0.5,),
                new_t!(inno = 2, weight = -0.5,),
                new_t!(inno = 3, weight = 1.0,),
            ],
            &[
                new_t!(inno = 5, weight = 0.5,),
                new_t!(inno = 6, weight = -0.5,),
            ],
        );
        assert_f64_approx!(diff, 0., "diff ne: {diff}, 0.")
    });

    test_t!(
    test_avg_param_diff_no_diff[T: WConnection | BWConnection]() {
        let diff = avg_param_diff(
            &[
                new_t!(inno = 1, weight = 0.5,),
                new_t!(inno = 2, weight = -0.5,),
                new_t!(inno = 3, weight = 1.0,),
            ],
            &[
                new_t!(inno = 1, weight = 0.5,),
                new_t!(inno = 2, weight = -0.5,),
                new_t!(inno = 3, weight = 1.0,),
            ],
        );
        assert_f64_approx!(diff, 0.0, "diff ne: {diff}, 0.");
    });

    test_t!(
    test_disjoint_excess_count[T: WConnection | BWConnection]() {
        assert_eq!(
            (4.0, 2.0),
            disjoint_excess_count(
                &[
                    new_t!(inno = 1),
                    new_t!(inno = 2),
                    new_t!(inno = 6),
                ],
                &[
                    new_t!(inno = 1),
                    new_t!(inno = 3),
                    new_t!(inno = 4),
                    new_t!(inno = 8),
                    new_t!(inno = 10),
                ]
            )
        );
    });

    test_t!(
    test_disjoint_excess_count_symmetrical[T: WConnection | BWConnection]() {
        let l = vec![
            new_t!(inno = 1),
            new_t!(inno = 2),
            new_t!(inno = 6),
        ];
        let r = vec![
            new_t!(inno = 1),
            new_t!(inno = 3),
            new_t!(inno = 4),
            new_t!(inno = 8),
            new_t!(inno = 10),
        ];
        assert_eq!(disjoint_excess_count(&l, &r), disjoint_excess_count(&r, &l));
    });

    test_t!(
    test_disjoint_excess_count_empty[T: WConnection | BWConnection]() {
        let full = vec![new_t!(inno = 1), new_t!(inno = 2)];
        assert_eq!((0.0, 2.0), disjoint_excess_count(&full, &[]));
        assert_eq!((0.0, 2.0), disjoint_excess_count(&[], &full));
        assert_eq!((0.0, 0.0), disjoint_excess_count::<T>(&[], &[]));
    });

    test_t!(
    test_disjoint_excess_count_hanging_l[T: WConnection | BWConnection]() {
        assert_eq!(
            (0.0, 1.0),
            disjoint_excess_count(
                &[
                    new_t!(inno = 0),
                    new_t!(inno = 1),
                    new_t!(inno = 2),
                ],
                &[new_t!(inno = 0), new_t!(inno = 1),]
            )
        )
    });

    test_t!(
    test_disjoint_excess_count_no_overlap[T: WConnection | BWConnection]() {
        assert_eq!(
            (2.0, 2.0),
            disjoint_excess_count(
                &[new_t!(inno = 1), new_t!(inno = 2),],
                &[new_t!(inno = 3), new_t!(inno = 4),]
            )
        );
    });

    test_t!(
    test_disjoint_excess_count_short_larger_inno[T: WConnection | BWConnection]() {
        assert_eq!(
            (3.0, 1.0),
            disjoint_excess_count(
                &[new_t!(inno = 10)],
                &[
                    new_t!(inno = 1),
                    new_t!(inno = 2),
                    new_t!(inno = 3),
                ]
            )
        );
    });

    fn assert_crossover_eq<C: Connection>(l: &[C], r: &[C]) {
        for (l, r) in [(l, r), (r, l)] {
            let l_map = l.iter().map(|c| (c.inno(), c)).collect::<HashMap<_, &_>>();
            let r_map = r.iter().map(|c| (c.inno(), c)).collect::<HashMap<_, &_>>();
            let inno = l_map
                .keys()
                .collect::<HashSet<_>>()
                .union(&r_map.keys().collect::<HashSet<_>>())
                .cloned()
                .cloned()
                .collect::<HashSet<_>>();

            let mut rng = default_rng();
            for _ in 0..1000 {
                let lr = crossover_eq(l, r, &mut rng);
                assert_eq!(inno.len(), lr.len());

                let lr_inno = lr.iter().map(|c| c.inno()).collect::<HashSet<_>>();
                assert!(inno.is_subset(&lr_inno));
                assert!(inno.is_superset(&lr_inno));
                assert!(lr.is_sorted_by_key(|c| c.inno()));
                for ref lr_conn in lr {
                    match (l_map.get(&lr_conn.inno()), r_map.get(&lr_conn.inno())) {
                        (None, None) => panic!("{} is in neither l nor r", lr_conn.inno()),
                        (None, Some(conn)) | (Some(conn), None) => {
                            assert_some_normalized!(lr_conn, [*conn]; {.enable()})
                        }
                        (Some(l_conn), Some(r_conn)) => {
                            assert_some_normalized!(lr_conn, [*l_conn, *r_conn]; {.enable()});
                        }
                    }
                }
            }
        }
    }

    test_t!(
    test_crossover_eq[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
            new_t!(inno = 2, from = 1_3),
        ];
        let r = [
            new_t!(inno = 0, from = 2_1),
            new_t!(inno = 2, from = 2_2),
            new_t!(inno = 3, from = 2_3),
        ];

        assert_crossover_eq(&l, &r);
    });

    test_t!(
    test_crossover_eq_empty[T: WConnection | BWConnection]() {
        let l = [new_t!(inno = 2, from = 1)];

        assert_crossover_eq(&l, &[]);
        assert_crossover_eq::<T>(&[], &[]);
    });

    test_t!(
    test_crossover_eq_overflow[T: WConnection | BWConnection]() {
        let l = [new_t!(inno = 0, from = 1_1)];
        let r = [new_t!(inno = 1, from = 2_1)];

        assert_crossover_eq(&l, &r);

        let l = [new_t!(inno = 1, from = 1_1)];
        let r = [new_t!(inno = 0, from = 2_1)];

        assert_crossover_eq(&l, &r);
    });

    test_t!(
    @panic test_crossover_eq_catchup_l[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
        ];
        let r = [new_t!(inno = 1, from = 2_1)];
        let mut rng = default_rng();
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng);
            assert_eq!(lr.len(), 2);
            assert_some_normalized!(&lr[0], [&l[0]]; {.enable()});
            assert_some_normalized!(&lr[1], [&r[0]]; {.enable()}, "not from r_0");
        }
    });

    test_t!(
    @panic test_crossover_eq_catchup_r[T: WConnection | BWConnection]() {
        let l = [new_t!(inno = 1, from = 2_1)];
        let r = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
        ];
        let mut rng = default_rng();
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng);
            assert_eq!(lr.len(), 2);
            assert_some_normalized!(&lr[0], [&r[0]]; {.enable()});
            assert_some_normalized!(&lr[1], [&l[0]]; {.enable()}, "not from l_0");
        }
    });

    test_t!(
    @panic test_crossover_eq_both_step_l[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
        ];
        let r = [
            new_t!(inno = 0, from = 2_1),
            new_t!(inno = 1, from = 2_2),
        ];
        let mut rng = default_rng();
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng);
            assert_eq!(lr.len(), 2);
            assert_some_normalized!(&lr[0], [&l[0], &r[0]]; {.enable()});
            assert_some_normalized!(&lr[1], [&l[1]]; {.enable()}, "not from l_1");
        }

    });

    test_t!(
    @panic test_crossover_eq_both_step_r[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
        ];
        let r = [
            new_t!(inno = 0, from = 2_1),
            new_t!(inno = 1, from = 2_2),
        ];
        let mut rng = default_rng();
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng);
            assert_eq!(lr.len(), 2);
            assert_some_normalized!(&lr[0], [&l[0], &r[0]]; {.enable()});
            assert_some_normalized!(&lr[1], [&r[1]]; {.enable()}, "not from r_1");
        }
    });

    fn assert_crossover_ne<C: Connection>(l: &[C], r: &[C]) {
        for (l, r) in [(l, r), (r, l)] {
            let l_map = l.iter().map(|c| (c.inno(), c)).collect::<HashMap<_, &_>>();
            let r_map = r.iter().map(|c| (c.inno(), c)).collect::<HashMap<_, &_>>();
            let l_keys = l_map.keys().cloned().collect::<HashSet<_>>();
            let inno = l_keys
                .union(&r_map.keys().cloned().collect::<HashSet<_>>())
                .cloned()
                .collect::<HashSet<_>>();

            let mut rng = default_rng();
            for _ in 0..1000 {
                let lr = crossover_ne(l, r, &mut rng);
                assert_eq!(lr.len(), l.len());

                let lr_inno = lr.iter().map(|c| c.inno()).collect::<HashSet<_>>();
                assert!(l_keys.is_subset(&lr_inno));
                assert!(l_keys.is_superset(&lr_inno));
                assert!(inno.is_superset(&lr_inno));
                assert!(lr.is_sorted_by_key(|c| c.inno()));
                for ref lr_conn in lr {
                    match (l_map.get(&lr_conn.inno()), r_map.get(&lr_conn.inno())) {
                        (None, None) => panic!("{} is in neither l nor r", lr_conn.inno()),
                        (None, Some(conn)) => panic!("{} is in only r", conn.inno()),
                        (Some(conn), None) => {
                            assert_some_normalized!(lr_conn, [*conn]; {.enable()})
                        }
                        (Some(l_conn), Some(r_conn)) => {
                            assert_some_normalized!(lr_conn, [*l_conn, *r_conn]; {.enable()})
                        }
                    }
                }
            }
        }
    }

    test_t!(
    test_crossover_ne[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
            new_t!(inno = 2, from = 1_3),
            new_t!(inno = 9, from = 1_4),
        ];
        let r = [
            new_t!(inno = 0, from = 2_1),
            new_t!(inno = 2, from = 2_2),
            new_t!(inno = 3, from = 2_3),
            new_t!(inno = 4, from = 2_4),
            new_t!(inno = 7, from = 2_5),
        ];

        assert_crossover_ne(&l, &r);
    });

    test_t!(
    test_crossover_ne_empty[T: WConnection | BWConnection]() {
        let l = [new_t!(inno = 0, from = 1_1)];

        assert_crossover_ne(&l, &[]);
        assert_crossover_ne::<T>(&[], &[]);
    });

    test_t!(
    test_crossover_ne_no_overlap[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 1, from = 1_1),
            new_t!(inno = 3, from = 1_2),
            new_t!(inno = 5, from = 1_3),
        ];
        let r = [
            new_t!(inno = 0, from = 2_1),
            new_t!(inno = 2, from = 2_2),
            new_t!(inno = 4, from = 2_3),
        ];

        assert_crossover_ne(&l, &r);
    });

    test_t!(
    test_crossover_ne_full_overlap[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 1, from = 1_1),
            new_t!(inno = 2, from = 1_2),
            new_t!(inno = 3, from = 1_3),
        ];
        let r = [
            new_t!(inno = 1, from = 2_1),
            new_t!(inno = 2, from = 2_2),
            new_t!(inno = 3, from = 2_3),
        ];

        assert_crossover_ne(&l, &r);
    });

    test_t!(
    test_crossover_ne_overflow[T: WConnection | BWConnection]() {
        let l = [new_t!(inno = 10, from = 1_1)];
        let r = [
            new_t!(inno = 1, from = 2_1),
            new_t!(inno = 2, from = 2_2),
        ];

        assert_crossover_ne(&l, &r);
    });

    test_t!(
    test_crossover_ne_no_lt[T: WConnection | BWConnection]() {
        let l = [new_t!(inno = 0, from = 1_1)];
        let r = [new_t!(inno = 10, from = 2_1)];

        assert_crossover_ne(&l, &r);
    });

    test_t!(
    test_crossover_lt[T: WConnection | BWConnection]() {
        let l = [
            new_t!(inno = 0, from = 1_1),
            new_t!(inno = 1, from = 1_2),
            new_t!(inno = 2, from = 1_3),
        ];
        let r = [
            new_t!(inno = 0, from = 2_1),
            new_t!(inno = 2, from = 2_2),
            new_t!(inno = 3, from = 2_3),
            new_t!(inno = 4, from = 2_4),
        ];

        let mut rng = default_rng();
        assert_crossover_ne(&l, &r);
        for (le, ge) in crossover(&l, &r, Ordering::Less, &mut rng)
            .iter()
            .zip(crossover_ne(&r, &l, &mut rng))
        {
            assert_eq!(le.inno(), ge.inno());
        }
    });
}
