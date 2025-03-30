use super::{node::CTRNode, Connection};
use crate::specie::InnoGen;
use serde::{Deserialize, Serialize};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CTRConnection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Connection for CTRConnection {
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;

    type Node = CTRNode;

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

    fn bisect(&mut self, center: usize, inno: &mut InnoGen) -> (Self, Self) {
        self.disable();
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

    fn param_diff(&self, other: &Self) -> f64 {
        // TODO add other ctrnn specific diffs when we have those fields available
        // theta, bias, weight
        (self.weight - other.weight).abs()
    }
}

impl Default for CTRConnection {
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

impl Hash for CTRConnection {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}
