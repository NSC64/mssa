//! Isolated study workers; see scripts/research_studies.py and docs/RESEARCH_STUDIES.md.
fn main() {
    if let Err(error) =
        pssa::interdiffusion::run_research_study(&std::env::args().skip(1).collect::<Vec<_>>())
    {
        eprintln!("research study: {error}");
        std::process::exit(1);
    }
}
