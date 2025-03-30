use super::{Node, NodeKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CTRNode {
    Sensory,
    Action,
    Bias(f64),
    Internal,
}

impl Node for CTRNode {
    fn new(kind: NodeKind) -> Self {
        match kind {
            NodeKind::Sensory => Self::Sensory,
            NodeKind::Action => Self::Action,
            NodeKind::Internal => Self::Internal,
            NodeKind::Bias => Self::Bias(1.),
        }
    }

    fn kind(&self) -> super::NodeKind {
        match self {
            Self::Sensory => NodeKind::Sensory,
            Self::Action => NodeKind::Action,
            Self::Bias(_) => NodeKind::Bias,
            Self::Internal => NodeKind::Internal,
        }
    }

    fn bias(&self) -> f64 {
        match self {
            Self::Bias(b) => *b,
            _ => 0.,
        }
    }
}
