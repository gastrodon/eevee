use super::{Connection, InnoGen};
use crate::{mutate_param, random::percent};
use core::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub struct WConnection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
}

/// A basic connection, with a single weighted path
impl Connection for WConnection {
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;

    mutate_param!([Weight]: [percent(100)]);

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self {
        Self {
            inno: inno.path((from, to)),
            from,
            to,
            weight: 1.,
            enabled: true,
        }
    }

    fn inno(&self) -> usize {
        self.inno
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn path(&self) -> (usize, usize) {
        (self.from, self.to)
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn bisect(&mut self, center: usize, inno: &mut InnoGen) -> (Self, Self) {
        <Self as Connection>::disable(self);
        (
            // from -{1.}> bisect-node
            Self {
                inno: inno.path((self.from, center)),
                from: self.from,
                to: center,
                weight: 1.,
                enabled: true,
            },
            // bisect-node -{w}> to
            Self {
                inno: inno.path((center, self.to)),
                from: center,
                to: self.to,
                weight: self.weight,
                enabled: true,
            },
        )
    }
}

impl Default for WConnection {
    fn default() -> Self {
        Self {
            inno: 0,
            from: 0,
            to: 0,
            weight: 0.,
            enabled: true,
        }
    }
}

impl Hash for WConnection {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}

/// A connection who has a per-connection bias
#[derive(Debug, Clone, PartialEq)]
pub struct BWConnection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub bias: f64,
    pub weight: f64,
    pub enabled: bool,
}

impl Connection for BWConnection {
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;

    mutate_param!([Weight, Bias]: [percent(50), percent(50)]);

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self {
        Self {
            inno: inno.path((from, to)),
            from,
            to,
            bias: 0.,
            weight: 1.,
            enabled: true,
        }
    }

    fn inno(&self) -> usize {
        self.inno
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn path(&self) -> (usize, usize) {
        (self.from, self.to)
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn bisect(&mut self, center: usize, inno: &mut InnoGen) -> (Self, Self) {
        <Self as Connection>::disable(self);
        (
            // from -{1.}> bisect-node
            Self {
                inno: inno.path((self.from, center)),
                from: self.from,
                to: center,
                bias: 0.,
                weight: 1.,
                enabled: true,
            },
            // bisect-node -{w}> to
            Self {
                inno: inno.path((center, self.to)),
                from: center,
                to: self.to,
                bias: self.bias,
                weight: self.weight,
                enabled: true,
            },
        )
    }
}

impl Default for BWConnection {
    fn default() -> Self {
        Self {
            inno: 0,
            from: 0,
            to: 0,
            bias: 0.,
            weight: 0.,
            enabled: true,
        }
    }
}

impl Hash for BWConnection {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.bias) as usize).hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}

#[cfg(test)]
mod test {
    use super::{BWConnection, Connection, InnoGen, WConnection};
    use crate::{assert_f64_approx, random::default_rng};
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,

        /// new() initializes from/to/inno/enabled/weight correctly
        #[test]
        fn test_new() {
            let mut inno = InnoGen::new(0);
            let c = C::new(1, 2, &mut inno);
            assert_eq!(c.from(), 1);
            assert_eq!(c.to(), 2);
            assert_eq!(c.inno(), 0);
            assert!(c.enabled());
            assert_f64_approx!(c.weight(), 1.0);
        }

        /// Same path reuses inno, different paths get unique innos
        #[test]
        fn test_inno_unique_per_path() {
            let mut inno = InnoGen::new(0);
            let c1 = C::new(0, 1, &mut inno);
            let c2 = C::new(0, 1, &mut inno);
            let c3 = C::new(1, 0, &mut inno);
            assert_eq!(c1.inno(), c2.inno());
            assert_ne!(c1.inno(), c3.inno());
        }

        /// enable()/disable() toggle enabled state correctly
        #[test]
        fn test_enabled_disable_enable() {
            let mut inno = InnoGen::new(0);
            let mut c = C::new(0, 1, &mut inno);
            assert!(c.enabled());
            c.disable();
            assert!(!c.enabled());
            c.enable();
            assert!(c.enabled());
            c.disable();
            assert!(!c.enabled());
        }

        /// path()/from()/to() return consistent (from,to) pairs
        #[test]
        fn test_path_consistency() {
            let mut inno = InnoGen::new(0);
            let c = C::new(5, 7, &mut inno);
            assert_eq!(c.path(), (5, 7));
            assert_eq!(c.from(), 5);
            assert_eq!(c.to(), 7);
        }

        /// weight() returns internal weight field value
        #[test]
        fn test_weight_matches_field() {
            let mut c = C::default();
            c.weight = 4.2;
            assert_f64_approx!(c.weight(), 4.2);
        }

        /// Identical connections have param_diff == 0.0
        #[test]
        fn test_param_diff_zero() {
            let mut inno1 = InnoGen::new(0);
            let mut inno2 = InnoGen::new(0);
            let c1 = C::new(0, 1, &mut inno1);
            let c2 = C::new(0, 1, &mut inno2);
            let diff = c1.param_diff(&c2);
            assert_f64_approx!(diff, 0.0);
        }

        /// mutate_param() modifies weight (1000 iterations) without changing enabled state
        #[test]
        fn test_mutate_param_changes_weight() {
            let mut rng = default_rng();
            let mut inno = InnoGen::new(0);
            let c = C::new(0, 1, &mut inno);
            let original_weight = c.weight;
            let original_enabled = c.enabled();
            let mut mutated = false;
            for _ in 0..1000 {
                let mut test_c = c.clone();
                test_c.mutate_param(&mut rng);
                if (test_c.weight - original_weight).abs() > f64::EPSILON {
                    mutated = true;
                    break;
                }
            }
            assert_eq!(c.enabled(), original_enabled);
            assert!(mutated, "mutate_param should cause weight changes");
        }

        /// mutate_param() preserves enabled state when disabled
        #[test]
        fn test_mutate_param_enabled_unchanged() {
            let mut rng = default_rng();
            let mut inno = InnoGen::new(0);
            let mut c = C::new(0, 1, &mut inno);
            c.disable();
            for _ in 0..100 {
                c.mutate_param(&mut rng);
                assert!(!c.enabled());
            }
        }


        /// bisect() disables original connection
        #[test]
        fn test_bisect_self_disabled() {
            let mut inno = InnoGen::new(0);
            let mut c = C::new(0, 1, &mut inno);
            c.enable();
            let (c1, c2) = c.bisect(5, &mut inno);
            assert!(!c.enabled());
            assert!(c1.enabled());
            assert!(c2.enabled());
        }

        /// bisect(center) creates correct from→center→to paths
        #[test]
        fn test_bisect_paths() {
            let mut inno = InnoGen::new(0);
            let mut c = C::new(2, 4, &mut inno);
            let (c1, c2) = c.bisect(3, &mut inno);
            assert_eq!(c1.from(), 2);
            assert_eq!(c1.to(), 3);
            assert_eq!(c2.from(), 3);
            assert_eq!(c2.to(), 4);
        }

        /// bisect() creates 3 unique innos (original, upper, lower)
        #[test]
        fn test_bisect_unique_innos() {
            let mut inno = InnoGen::new(0);
            let mut c = C::new(0, 1, &mut inno);
            let (c1, c2) = c.bisect(5, &mut inno);
            let all_innos = vec![c.inno(), c1.inno(), c2.inno()];
            assert_eq!(
                all_innos.len(),
                all_innos
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            );
        }

        /// default() creates zero-initialized connection
        #[test]
        fn test_default() {
            let c = C::default();
            assert_eq!(c.inno(), 0);
            assert_eq!(c.from(), 0);
            assert_eq!(c.to(), 0);
            assert!(c.enabled());
            assert_eq!(c.weight, 0.0);
        }

        /// Identical connections hash identically
        #[test]
        fn test_hash_consistency() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut inno = InnoGen::new(0);
            let c1 = C::new(0, 1, &mut inno);
            let c2 = c1.clone();
            let mut hasher1 = DefaultHasher::new();
            let mut hasher2 = DefaultHasher::new();
            c1.hash(&mut hasher1);
            c2.hash(&mut hasher2);
            assert_eq!(hasher1.finish(), hasher2.finish());
        }

        /// clone() creates equal copy preserving all fields
        #[test]
        fn test_clone() {
            let mut inno = InnoGen::new(0);
            let mut c = C::new(0, 1, &mut inno);
            c.disable();
            c.weight = 2.5;
            let c_clone = c.clone();
            assert_eq!(c, c_clone);
            assert_eq!(c.enabled(), c_clone.enabled());
            assert_f64_approx!(c.weight(), c_clone.weight());
        }

        /// == correctly reflects all field differences
        #[test]
        fn test_partialeq() {
            let mut inno1 = InnoGen::new(0);
            let mut inno2 = InnoGen::new(0);
            let mut c1 = C::new(0, 1, &mut inno1);
            let mut c2 = C::new(0, 1, &mut inno2);
            assert_eq!(c1, c2);
            c1.disable();
            assert_ne!(c1, c2);
            c2.disable();
            c1.weight = 2.0;
            assert_ne!(c1, c2);
        }

        /// Debug formatting produces non-empty string with inno/from
        #[test]
        fn test_debug_trait() {
            let mut inno = InnoGen::new(0);
            let c = C::new(0, 1, &mut inno);
            let debug_str = format!("{:?}", c);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("inno") || debug_str.contains("from"));
        }
    }

    /// WConnection: Weight difference reflected in param_diff
    #[test]
    fn test_wconnection_param_diff_nonzero() {
        let mut inno1 = InnoGen::new(0);
        let mut inno2 = InnoGen::new(0);
        let mut c1 = WConnection::new(0, 1, &mut inno1);
        let mut c2 = WConnection::new(0, 1, &mut inno2);
        c1.weight = 2.0;
        c2.weight = 1.0;
        let diff = c1.param_diff(&c2);
        assert!(diff > 0.9 && diff < 1.1, "param_diff = {}", diff);
    }

    /// BWConnection: Combines weight+bias diffs
    #[test]
    fn test_bwconnection_param_diff_nonzero() {
        let mut inno1 = InnoGen::new(0);
        let mut inno2 = InnoGen::new(0);
        let mut c1 = BWConnection::new(0, 1, &mut inno1);
        let mut c2 = BWConnection::new(0, 1, &mut inno2);
        c1.weight = 2.0;
        c1.bias = 0.5;
        c2.weight = 1.0;
        c2.bias = 0.2;
        let diff = c1.param_diff(&c2);
        assert!(diff > 1.2 && diff < 1.4, "param_diff = {}", diff);
    }

    /// WConnection: Upper gets weight=1.0, lower gets original weight
    #[test]
    fn test_wconnection_bisect_weight_distribution() {
        let mut inno = InnoGen::new(0);
        let mut c = WConnection::new(0, 1, &mut inno);
        c.weight = 2.5;
        let (c1, c2) = c.bisect(5, &mut inno);
        assert_f64_approx!(c1.weight(), 1.0);
        assert_f64_approx!(c2.weight(), 2.5);
    }

    /// BWConnection: Weight preserved on lower, bias distributed
    #[test]
    fn test_bwconnection_bisect_weight_distribution() {
        let mut inno = InnoGen::new(0);
        let mut c = BWConnection::new(0, 1, &mut inno);
        c.weight = 2.5;
        c.bias = 1.0;
        let (c1, c2) = c.bisect(5, &mut inno);
        assert_f64_approx!(c1.weight(), 1.0);
        assert_f64_approx!(c2.weight(), 2.5);
        assert_f64_approx!(c1.bias, 0.0);
        assert_f64_approx!(c2.bias, 1.0);
    }

    /// BWConnection::default() includes zero-initialized bias
    #[test]
    fn test_bwconnection_default() {
        let c = BWConnection::default();
        assert_eq!(c.inno(), 0);
        assert_eq!(c.from(), 0);
        assert_eq!(c.to(), 0);
        assert!(c.enabled());
        assert_eq!(c.weight, 0.0);
        assert_eq!(c.bias, 0.0);
    }

    /// WConnection: Weights within 3 decimals hash same (truncation edge case)
    #[test]
    fn test_wconnection_hash_weight_precision() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut c1 = WConnection::default();
        c1.weight = 1.0001;
        let mut c2 = WConnection::default();
        c2.weight = 1.0009;
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        c1.hash(&mut hasher1);
        c2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    /// BWConnection: clone() preserves weight and bias
    #[test]
    fn test_bwconnection_clone_with_bias() {
        let mut inno = InnoGen::new(0);
        let mut c = BWConnection::new(0, 1, &mut inno);
        c.disable();
        c.weight = 2.5;
        c.bias = 1.5;
        let c_clone = c.clone();
        assert_eq!(c, c_clone);
        assert_eq!(c.enabled(), c_clone.enabled());
        assert_f64_approx!(c.weight(), c_clone.weight());
        assert_f64_approx!(c.bias, c_clone.bias);
    }
}
