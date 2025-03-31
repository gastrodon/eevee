use super::{Node, NodeKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BNode {
    bias: f64,
    kind: NodeKind,
}

impl Node for BNode {
    fn new(kind: NodeKind) -> Self {
        Self { bias: 0., kind }
    }

    fn kind(&self) -> NodeKind {
        self.kind
    }

    fn bias(&self) -> f64 {
        self.bias
    }

    fn mutate_params(&mut self, rng: &mut (impl rand::RngCore + crate::Happens)) {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NonBNode {
    Sensory,
    Action,
    Internal,
    Static(f64),
}

impl Node for NonBNode {
    fn new(kind: NodeKind) -> Self {
        match kind {
            NodeKind::Sensory => Self::Sensory,
            NodeKind::Action => Self::Action,
            NodeKind::Internal => Self::Internal,
            NodeKind::Static => Self::Static(1.),
        }
    }

    fn kind(&self) -> super::NodeKind {
        match self {
            Self::Sensory => NodeKind::Sensory,
            Self::Action => NodeKind::Action,
            Self::Internal => NodeKind::Internal,
            Self::Static(_) => NodeKind::Static,
        }
    }

    fn bias(&self) -> f64 {
        match self {
            Self::Static(b) => *b,
            _ => 0.,
        }
    }

    fn mutate_params(&mut self, rng: &mut (impl rand::RngCore + crate::Happens)) {
        todo!()
    }
}
