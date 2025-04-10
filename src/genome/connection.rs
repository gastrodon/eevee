use super::{Connection, InnoGen};
use crate::{mutate_param, random::percent};
use core::hash::Hash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
