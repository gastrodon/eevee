use eevee::{
    genome::nn_policies::Recurrent,
    genome::{InnoGen, WConnection},
    tui::NeuralNetFlow,
    Genome,
};
use ratatui::{backend::TestBackend, Terminal};

fn print_genome(
    label: &str,
    genome: &Recurrent<WConnection>,
    max_height: usize,
    render_height: u16,
) {
    let flow = NeuralNetFlow::<WConnection, _>::new(genome, max_height);
    let h = render_height;
    let w = flow.required_width(h).max(1);

    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| {
            f.render_widget(
                NeuralNetFlow::<WConnection, _>::new(genome, max_height),
                f.size(),
            )
        })
        .unwrap();

    println!("=== {} ({}x{}) ===", label, w, h);
    let buf = terminal.backend().buffer();
    for y in (0..h).rev() {
        for x in 0..w {
            print!("{}", buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
        }
        println!();
    }
    println!();
}

fn main() {
    let rng = &mut eevee::random::default_rng();

    // 3-input, 2-output, no hidden
    let (genome_3x2, _) = Recurrent::<WConnection>::new(3, 2);
    print_genome("3x2 no hidden", &genome_3x2, 4, 20);

    // 2-input, 1-output, 5 hidden nodes
    let (mut genome_hidden, head) = Recurrent::<WConnection>::new(2, 1);
    let mut inno = InnoGen::new(head);
    for _ in 0..5 {
        genome_hidden.bisect_connection(rng, &mut inno).unwrap();
    }
    print_genome("2x1 + 5 hidden", &genome_hidden, 4, 20);

    // 1-input, 1-output, 9 hidden → 3 hidden columns at max_height=3
    let (mut genome_wide, head) = Recurrent::<WConnection>::new(1, 1);
    let mut inno = InnoGen::new(head);
    for _ in 0..9 {
        genome_wide.bisect_connection(rng, &mut inno).unwrap();
    }
    print_genome("1x1 + 9 hidden (max_height=3)", &genome_wide, 3, 20);
}
