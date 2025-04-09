use super::{Biased, Node, NodeKind};
use crate::{node, random::percent};
use rand::RngCore;
use serde::{Deserialize, Serialize};

node!(BNode, [bias]: [percent(100)]);
node!(BTNode, [bias, timescale]: [percent(50), percent(50)]);

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

    fn param_diff(&self, _: &Self) -> f64 {
        0.
    }

    fn mutate_param(&mut self, _: &mut impl RngCore) {}
}

/// while this seems contradictory, we treat the static component of a nodes as its bias
/// the rest of the nodes always have a bias of 0, being non-biased
impl Biased for NonBNode {
    fn bias(&self) -> f64 {
        match self {
            Self::Static(b) => *b,
            _ => 0.,
        }
    }
}
