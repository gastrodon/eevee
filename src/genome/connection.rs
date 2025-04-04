use super::{Connection, Node};
use crate::{
    mutate_params,
    random::{percent, EventKind},
    specie::InnoGen,
    Happens,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{hash::Hash, marker::PhantomData};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WConnection<N: Node> {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
    _phantom: PhantomData<N>,
}

impl<N: Node> Connection<N> for WConnection<N> {
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;

    mutate_params!(WParam[Weight]: [percent(100)]);

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self {
        Self {
            inno: inno.path((from, to)),
            from,
            to,
            weight: 1.,
            enabled: true,
            _phantom: PhantomData,
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
        <Self as Connection<N>>::disable(self);
        (
            // from -{1.}> bisect-node
            Self {
                inno: inno.path((self.from, center)),
                from: self.from,
                to: center,
                weight: 1.,
                enabled: true,
                _phantom: PhantomData,
            },
            // bisect-node -{w}> to
            Self {
                inno: inno.path((center, self.to)),
                from: center,
                to: self.to,
                weight: self.weight,
                enabled: true,
                _phantom: PhantomData,
            },
        )
    }

    fn param_diff(&self, other: &Self) -> f64 {
        // TODO add other ctrnn specific diffs when we have those fields available
        // theta, bias, weight
        (self.weight - other.weight).abs()
    }
}

impl<N: Node> Default for WConnection<N> {
    fn default() -> Self {
        Self {
            inno: 0,
            from: 0,
            to: 0,
            weight: 0.,
            enabled: true,
            _phantom: PhantomData,
        }
    }
}

impl<N: Node> Hash for WConnection<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BWConnection<N: Node> {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub bias: f64,
    pub weight: f64,
    pub enabled: bool,
    _phantom: PhantomData<N>,
}

impl<N: Node> Connection<N> for BWConnection<N> {
    const EXCESS_COEFFICIENT: f64 = 1.0;
    const DISJOINT_COEFFICIENT: f64 = 1.0;
    const PARAM_COEFFICIENT: f64 = 0.4;

    mutate_params!(BWParam[Weight, Bias]: [percent(50), percent(50)]);

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self {
        Self {
            inno: inno.path((from, to)),
            from,
            to,
            bias: 0.,
            weight: 1.,
            enabled: true,
            _phantom: PhantomData,
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
        <Self as Connection<N>>::disable(self);
        (
            // from -{1.}> bisect-node
            Self {
                inno: inno.path((self.from, center)),
                from: self.from,
                to: center,
                bias: 0.,
                weight: 1.,
                enabled: true,
                _phantom: PhantomData,
            },
            // bisect-node -{w}> to
            Self {
                inno: inno.path((center, self.to)),
                from: center,
                to: self.to,
                bias: self.bias,
                weight: self.weight,
                enabled: true,
                _phantom: PhantomData,
            },
        )
    }

    fn param_diff(&self, other: &Self) -> f64 {
        (self.bias - other.bias).abs() + (self.weight - other.weight).abs()
    }
}

impl<N: Node> Default for BWConnection<N> {
    fn default() -> Self {
        Self {
            inno: 0,
            from: 0,
            to: 0,
            bias: 0.,
            weight: 0.,
            enabled: true,
            _phantom: PhantomData,
        }
    }
}

impl<N: Node> Hash for BWConnection<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.bias) as usize).hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}
