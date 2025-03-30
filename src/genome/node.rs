use super::{Node, NodeKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CTRNode {
    Sensory,
    Action,
    Internal,
    Static(f64),
}

impl Node for CTRNode {
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
}
