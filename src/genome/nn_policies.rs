use super::{
    nn_organism::{NNOrganism, PathPolicy},
    Connection,
};
use fxhash::FxHashSet;

/// Topology policy that allows all connections, including recurrent ones.
#[derive(Clone, Debug, Default)]
pub struct RecurrentPolicy;

impl<C: Connection> PathPolicy<C> for RecurrentPolicy {
    fn allows(_from: usize, _to: usize, _connections: &[C]) -> bool {
        true
    }
}

/// A genome that allows recurrent connections.
pub type Recurrent<C> = NNOrganism<C, RecurrentPolicy>;

impl<C: Connection> crate::Recurrent for NNOrganism<C, RecurrentPolicy> {}

/// Topology policy that rejects connections that would form a directed cycle.
#[derive(Clone, Debug, Default)]
pub struct NonRecurrentPolicy;

fn would_create_cycle<C: Connection>(from: usize, to: usize, connections: &[C]) -> bool {
    if from == to {
        return true;
    }
    // DFS from `to`; if we reach `from`, adding this edge creates a cycle.
    let mut visited = FxHashSet::default();
    let mut stack = vec![to];
    while let Some(node) = stack.pop() {
        if node == from {
            return true;
        }
        if visited.insert(node) {
            for c in connections {
                if c.from() == node {
                    stack.push(c.to());
                }
            }
        }
    }
    false
}

impl<C: Connection> PathPolicy<C> for NonRecurrentPolicy {
    fn allows(from: usize, to: usize, connections: &[C]) -> bool {
        !would_create_cycle(from, to, connections)
    }
}

/// A genome that only allows feedforward (acyclic) connections.
pub type NonRecurrent<C> = NNOrganism<C, NonRecurrentPolicy>;

impl<C: Connection> crate::Forward for NNOrganism<C, NonRecurrentPolicy> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        genome::{connection::BWConnection, Genome, InnoGen, WConnection},
        random::default_rng,
    };
    use eevee_macros::fn_matrix;

    fn_matrix! {
        C: WConnection | BWConnection,
        G: Recurrent<C> | NonRecurrent<C>,

        /// open_path generates valid connections
        #[test]
        fn test_gen_connection() {
            let (mut genome, _) = G::new(1, 1);
            genome.connections = vec![];

            for _ in 0..100 {
                match genome.open_path(&mut default_rng()) {
                    Some((0, 1)) => {}, // sensory -> action
                    Some(p) => unreachable!("invalid pair {p:?} gen'd"),
                    None => unreachable!("no path gen'd"),
                }
            }

            genome.push_connection(C::new(2, 1, &mut InnoGen::new(0)));
            for _ in 0..100 {
                assert_eq!(genome.open_path(&mut default_rng()), Some((0, 1)));
            }
        }

        /// empty genome has no open paths
        #[test]
        fn test_gen_connection_none_possible() {
            let (genome, _) = G::new(0, 0);
            assert_eq!(genome.open_path(&mut default_rng()), None);
        }

        /// new_connection adds connections with unique innovation IDs
        #[test]
        fn test_mutate_connection() {
            let (mut genome, _) = G::new(4, 4);
            let mut inno = InnoGen::new(0);
            genome.connections = vec![];
            genome.push_connection(C::new(0, 1, &mut inno));
            genome.push_connection(C::new(1, 2, &mut inno));

            let before = genome.clone();
            genome.new_connection(&mut default_rng(), &mut inno).unwrap_or_else(|e| panic!("failed new_connection: {e}"));

            assert_eq!(genome.connections().len(), before.connections().len() + 1);

            let tail = genome.connections().last().unwrap();
            assert!(!before.connections().iter().any(|c| c.inno() == tail.inno()));
            assert!(!before.connections().iter().any(|c| c.path() == tail.path()));
            assert_eq!(tail.weight(), 1.);
        }
    }

    fn_matrix! {
        C: WConnection | BWConnection,
        G: NonRecurrent<C>,

        /// NonRecurrent prevents cycle-forming paths
        #[test]
        fn test_no_cycles() {
            // nodes: sensory(0), action(1), internal(2), internal(3)
            // connections: 2->3
            // open_path must never return (3, 2) — that would close a cycle
            let (mut genome, _) = G::new(1, 1);
            genome.connections = vec![];
            genome.push_node(); // 2
            genome.push_node(); // 3
            genome.push_connection(C::new(2, 3, &mut InnoGen::new(0)));

            for _ in 0..200 {
                if let Some((from, to)) = genome.open_path(&mut default_rng()) {
                    assert!(
                        !(from == 3 && to == 2),
                        "open_path returned cycle-forming path (3, 2)"
                    );
                }
            }
        }
    }
}
