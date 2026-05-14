use std::marker::PhantomData;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Color,
    symbols::Marker,
    widgets::{
        canvas::{Canvas, Circle, Line},
        Widget,
    },
};

use crate::genome::{Connection, Genome};

/// A group of `count` neurons laid out in a grid with a fixed maximum column height.
///
/// Width expands with the number of neurons: `ceil(count / max_height)` columns,
/// each `col_width` logical units wide. Height stays constant at `max_height` rows.
///
/// Positions are returned in canvas logical coordinates given an x offset and canvas height.
#[derive(Clone)]
pub struct NeuronGroup {
    pub count: usize,
    pub max_height: usize,
}

impl NeuronGroup {
    pub fn new(count: usize, max_height: usize) -> Self {
        Self { count, max_height }
    }

    /// Number of columns needed to fit `count` neurons at `max_height` per column.
    pub fn num_cols(&self) -> usize {
        if self.count == 0 {
            1
        } else {
            self.count.div_ceil(self.max_height)
        }
    }

    /// Logical width consumed by this group (in canvas units, same scale as canvas height).
    pub fn logical_width(&self, canvas_height: f64) -> f64 {
        self.num_cols() as f64 * canvas_height / self.max_height as f64
    }

    /// Positions for all neurons in this group, given the x coordinate of the group's
    /// left edge and the full canvas height.
    ///
    /// Returns one `(x, y)` per neuron, in node-index order.
    pub fn positions(&self, x_start: f64, group_width: f64, canvas_height: f64) -> Vec<(f64, f64)> {
        let num_cols = self.num_cols();
        (0..self.count)
            .map(|slot| {
                let col = slot / self.max_height;
                let row = slot % self.max_height;
                let col_size = if col == num_cols - 1 && self.count % self.max_height != 0 {
                    self.count % self.max_height
                } else {
                    self.max_height
                };
                let x = x_start + (col as f64 + 0.5) * group_width / num_cols as f64;
                let y = canvas_height * (row + 1) as f64 / (col_size + 1) as f64;
                (x, y)
            })
            .collect()
    }
}

/// Renders a genome as a horizontal flow of three [`NeuronGroup`]s:
/// inputs | hidden | outputs, with connections drawn across the full canvas.
///
/// Height is fixed at `max_height` nodes per column. Width grows with hidden node count.
/// Use [`NeuralNetFlow::required_width`] to size the containing [`Rect`] before rendering.
pub struct NeuralNetFlow<'a, C: Connection, G: Genome<C>> {
    genome: &'a G,
    max_height: usize,
    _c: PhantomData<C>,
}

impl<'a, C: Connection, G: Genome<C>> NeuralNetFlow<'a, C, G> {
    pub fn new(genome: &'a G, max_height: usize) -> Self {
        Self {
            genome,
            max_height,
            _c: PhantomData,
        }
    }

    fn groups(&self) -> (NeuronGroup, NeuronGroup, NeuronGroup) {
        let sensory = self.genome.sensory();
        let action = self.genome.action();
        let hidden_count = self.genome.node_count().saturating_sub(action.end);
        (
            NeuronGroup::new(sensory.len(), self.max_height),
            NeuronGroup::new(hidden_count, self.max_height),
            NeuronGroup::new(action.len(), self.max_height),
        )
    }

    /// Required canvas width in character columns given a canvas height in rows.
    pub fn required_width(&self, canvas_height: u16) -> u16 {
        let h = canvas_height as f64;
        let (inputs, hidden, outputs) = self.groups();
        let total =
            inputs.logical_width(h) + hidden.logical_width(h) + outputs.logical_width(h);
        total.ceil() as u16
    }
}

impl<'a, C: Connection, G: Genome<C>> Widget for NeuralNetFlow<'a, C, G> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let w = area.width as f64;
        let h = area.height as f64;

        let (input_group, hidden_group, output_group) = self.groups();

        let input_w = input_group.logical_width(h);
        let hidden_w = hidden_group.logical_width(h);
        let output_w = output_group.logical_width(h);

        let input_positions = input_group.positions(0.0, input_w, h);
        let hidden_positions = hidden_group.positions(input_w, hidden_w, h);
        let output_positions = output_group.positions(input_w + hidden_w, output_w, h);

        // Build a flat lookup: node index → canvas (x, y).
        // Node layout: [0..sensory) inputs, [sensory..sensory+action) outputs,
        // [sensory+action..) hidden — but we render inputs | hidden | outputs left-to-right,
        // so we remap: sensory nodes first, then hidden, then action.
        let sensory = self.genome.sensory();
        let action = self.genome.action();
        let hidden_start = action.end;
        let n = self.genome.node_count();

        let mut positions = vec![(0.0f64, 0.0f64); n];
        for (slot, &pos) in input_positions.iter().enumerate() {
            positions[sensory.start + slot] = pos;
        }
        for (slot, &pos) in hidden_positions.iter().enumerate() {
            positions[hidden_start + slot] = pos;
        }
        for (slot, &pos) in output_positions.iter().enumerate() {
            positions[action.start + slot] = pos;
        }

        let genome = self.genome;
        Canvas::default()
            .marker(Marker::Braille)
            .x_bounds([0.0, w])
            .y_bounds([0.0, h])
            .paint(move |ctx| {
                for conn in genome.connections() {
                    if !conn.enabled() {
                        continue;
                    }
                    let (x1, y1) = positions[conn.from()];
                    let (x2, y2) = positions[conn.to()];
                    let color = if conn.weight() >= 0.0 {
                        Color::Cyan
                    } else {
                        Color::Red
                    };
                    ctx.draw(&Line { x1, y1, x2, y2, color });
                }
                for (i, &(x, y)) in positions.iter().enumerate() {
                    let color = if sensory.contains(&i) {
                        Color::Green
                    } else if action.contains(&i) {
                        Color::Yellow
                    } else {
                        Color::White
                    };
                    ctx.draw(&Circle { x, y, radius: 0.5, color });
                }
            })
            .render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{nn_policies::Recurrent, InnoGen, WConnection};
    use crate::random::default_rng;
    use ratatui::{backend::TestBackend, Terminal};

    fn render_genome(genome: &Recurrent<WConnection>, max_height: usize) {
        let flow = NeuralNetFlow::<WConnection, _>::new(genome, max_height);
        let w = flow.required_width(max_height as u16);
        let h = max_height as u16;
        let backend = TestBackend::new(w.max(1), h.max(1));
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                f.render_widget(
                    NeuralNetFlow::<WConnection, _>::new(genome, max_height),
                    f.size(),
                )
            })
            .unwrap();
    }

    #[test]
    fn test_neuron_group_positions_single_col() {
        let g = NeuronGroup::new(3, 8);
        assert_eq!(g.num_cols(), 1);
        let pos = g.positions(0.0, 10.0, 24.0);
        assert_eq!(pos.len(), 3);
        // all in the same x column
        assert!((pos[0].0 - pos[1].0).abs() < 1e-9);
        // y values evenly spaced: 24*1/4, 24*2/4, 24*3/4
        assert!((pos[0].1 - 6.0).abs() < 1e-9);
        assert!((pos[1].1 - 12.0).abs() < 1e-9);
        assert!((pos[2].1 - 18.0).abs() < 1e-9);
    }

    #[test]
    fn test_neuron_group_two_cols() {
        let g = NeuronGroup::new(5, 3);
        assert_eq!(g.num_cols(), 2);
        let pos = g.positions(0.0, 12.0, 24.0);
        // first 3 in col 0, last 2 in col 1
        assert_eq!(pos.len(), 5);
        assert!(pos[0].0 < pos[3].0); // col 0 left of col 1
    }

    #[test]
    fn test_required_width_grows_with_hidden() {
        let (mut genome, head) = Recurrent::<WConnection>::new(1, 1);
        let mut inno = InnoGen::new(head);
        let w0 =
            NeuralNetFlow::<WConnection, _>::new(&genome, 4).required_width(4);
        for _ in 0..5 {
            genome
                .bisect_connection(&mut default_rng(), &mut inno)
                .unwrap();
        }
        let w1 =
            NeuralNetFlow::<WConnection, _>::new(&genome, 4).required_width(4);
        assert!(w1 > w0, "width should grow as hidden nodes are added");
    }

    #[test]
    fn test_render_basic() {
        let (genome, _) = Recurrent::<WConnection>::new(3, 2);
        render_genome(&genome, 8);
    }

    #[test]
    fn test_render_with_hidden() {
        let (mut genome, head) = Recurrent::<WConnection>::new(2, 1);
        let mut inno = InnoGen::new(head);
        genome.bisect_connection(&mut default_rng(), &mut inno).unwrap();
        genome.bisect_connection(&mut default_rng(), &mut inno).unwrap();
        render_genome(&genome, 4);
    }

    #[test]
    fn test_render_empty() {
        let (genome, _) = Recurrent::<WConnection>::new(0, 0);
        render_genome(&genome, 4);
    }
}
