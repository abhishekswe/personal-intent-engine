use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "pie",
    about = "Personal Intent Engine - Intelligent AI middleware"
)]
struct Args {
    /// Text input to process
    #[arg(trailing_var_arg = true)]
    input: Vec<String>,

    /// Optimization mode: "direct", "enhanced", or "auto" (select from input
    /// complexity)
    #[arg(short, long, default_value = "auto")]
    mode: String,

    /// Translate spoken code patterns into syntax ("console dot log" ->
    /// "console.log(") after pronunciation correction. Off by default.
    #[arg(long)]
    code_mode: bool,

    /// LLM provider
    #[arg(short, long, default_value = "openai")]
    provider: String,

    /// Model name
    #[arg(long)]
    model: Option<String>,

    /// Verbose output (show intent, optimized prompt, etc.)
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let text = resolve_input(&args)?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut engine = pie_engine::PieEngine::new().await?;
        engine.set_code_mode(args.code_mode);

        if args.verbose {
            println!("[PIE] Mode: {}", args.mode);
            println!("[PIE] Provider: {}", args.provider);
            println!("[PIE] Input: {text}\n");
        }

        let result = engine.process(&text, &args.mode).await?;

        if args.verbose {
            println!("[PIE] Detected intent:");
            println!("  Objective: {}", result.intent.objective);
            println!("  Type: {:?}", result.intent.conversation_type);
            println!("  Confidence: {:?}", result.intent.confidence);
            println!("  Context: {:?}", result.intent.context);
            println!("  Constraints: {:?}", result.intent.constraints);
            println!();
            println!(
                "[PIE] Optimized prompt ({} chars):",
                result.optimized_prompt.len()
            );
            println!("{}", result.optimized_prompt);
            println!();
        }

        // Send to LLM
        let response = engine
            .send_to_llm(
                &result.optimized_prompt,
                &args.provider,
                args.model.as_deref(),
            )
            .await?;

        println!("{}", response);
        Ok(())
    })
}

/// Resolve the text input: CLI arguments, or stdin when none are given.
/// Voice is handled by the desktop app (on-device Moonshine STT), not the CLI.
fn resolve_input(args: &Args) -> anyhow::Result<String> {
    let text = if args.input.is_empty() {
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer)?;
        buffer.trim().to_string()
    } else {
        args.input.join(" ")
    };

    if text.is_empty() {
        anyhow::bail!("No input. Pass text as arguments or pipe it via stdin.");
    }
    Ok(text)
}
