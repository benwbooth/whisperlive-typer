mod audio;
mod client;
mod commands;
mod config;
mod grapheme;
mod mute;
mod protocol;
mod server;
mod typer;
mod websocket;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use config::Config;

#[derive(Parser)]
#[command(name = "whisper-typer", about = "Speech-to-text typing via WhisperLive")]
struct Cli {
    /// Config file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// WhisperLive server host
    #[arg(short = 'H', long)]
    host: Option<String>,

    /// WhisperLive server port
    #[arg(short, long)]
    port: Option<u16>,

    /// Language code
    #[arg(short, long)]
    language: Option<String>,

    /// Whisper model size
    #[arg(short, long)]
    model: Option<String>,

    /// Audio input device: index number or name substring
    #[arg(short, long)]
    device: Option<String>,

    /// List available audio input devices and exit
    #[arg(short = 'L', long)]
    list_devices: bool,

    /// ydotool socket path
    #[arg(short, long)]
    socket: Option<String>,

    /// Don't actually type, just log
    #[arg(short = 'n', long)]
    dry_run: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// VAD onset threshold (0.0-1.0)
    #[arg(long)]
    vad_onset: Option<f64>,

    /// VAD offset threshold (0.0-1.0)
    #[arg(long)]
    vad_offset: Option<f64>,

    /// Drop segments with no_speech_prob above this threshold
    #[arg(long)]
    no_speech_thresh: Option<f64>,

    /// Drop segments with avg_logprob below this threshold
    #[arg(long)]
    min_avg_logprob: Option<f64>,

    /// Don't auto-start the server if not running
    #[arg(long)]
    no_auto_start: bool,

    /// Directory containing docker-compose.yml for server auto-start
    #[arg(long)]
    server_dir: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .init();

    // Handle --list-devices
    if cli.list_devices {
        let devices = audio::list_devices()?;
        println!("Available audio input devices:");
        println!("{}", "-".repeat(60));
        for dev in &devices {
            println!("  [{}] \"{}\"", dev.index, dev.name);
            println!("      Sample Rate: {} Hz", dev.sample_rate);
        }
        println!("{}", "-".repeat(60));
        println!("Use --device \"<name>\" or --device <index>");
        return Ok(());
    }

    // Load config
    let mut config = Config::load(cli.config.as_deref());

    // CLI overrides
    if let Some(v) = cli.host { config.host = v; }
    if let Some(v) = cli.port { config.port = v; }
    if let Some(v) = cli.language { config.language = v; }
    if let Some(v) = cli.model { config.model = v; }
    if let Some(v) = cli.vad_onset { config.vad_onset = v; }
    if let Some(v) = cli.vad_offset { config.vad_offset = v; }
    if let Some(v) = cli.no_speech_thresh { config.no_speech_thresh = v; }
    if let Some(v) = cli.min_avg_logprob { config.min_avg_logprob = v; }
    if cli.no_auto_start { config.auto_start_server = false; }
    if let Some(v) = cli.server_dir { config.server_dir = v; }

    // Resolve audio device
    let device_str = cli.device.or_else(|| {
        if config.device.is_empty() {
            None
        } else {
            Some(config.device.clone())
        }
    });
    let device = if let Some(ref query) = device_str {
        Some(audio::find_device(query)?)
    } else {
        None
    };

    // Build platform typer and notifier
    // dry-run only affects typing, not notifications (matching Python behavior)
    let (typer_box, notifier_box): (Box<dyn typer::platform::Typer>, Box<dyn typer::platform::Notifier>) =
        if cli.dry_run {
            let (_, notifier) = build_platform_typer_notifier(cli.socket, &config);
            (Box::new(typer::dry_run::DryRunTyper), notifier)
        } else {
            build_platform_typer_notifier(cli.socket, &config)
        };

    let stateful_typer = Arc::new(typer::StatefulTyper::new(typer_box));

    // Print banner
    println!("{}", "=".repeat(50));
    println!("WhisperLive Typer (Rust)");
    println!("{}", "=".repeat(50));
    println!("Server: {}:{}", config.host, config.port);
    println!("Language: {}", config.language);
    println!("Model: {}", config.model);
    println!(
        "VAD: onset={}, offset={}",
        config.vad_onset, config.vad_offset
    );
    println!(
        "Filters: no_speech_thresh={}, min_avg_logprob={}",
        config.no_speech_thresh, config.min_avg_logprob
    );
    if let Some(ref q) = device_str {
        println!("Microphone: {q}");
    } else {
        println!("Microphone: System default");
    }
    if cli.dry_run {
        println!("Mode: DRY RUN (not typing)");
    }
    println!("Press Ctrl+C to stop");
    println!("{}", "=".repeat(50));

    // Run async
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // Ensure server is running
        if !server::ensure_server_running(&config).await {
            eprintln!("Error: Server is not running and could not be started.");
            eprintln!(
                "Make sure the WhisperLive server is running at {}:{}",
                config.host, config.port
            );
            eprintln!("Or use --no-auto-start to disable auto-start and see the full error.");
            std::process::exit(1);
        }

        let client = client::Client::new(config, stateful_typer, notifier_box, device);
        if let Err(e) = client.run().await {
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
    });

    Ok(())
}

#[cfg(target_os = "linux")]
fn build_platform_typer_notifier(
    socket: Option<String>,
    config: &Config,
) -> (Box<dyn typer::platform::Typer>, Box<dyn typer::platform::Notifier>) {
    let typer = typer::linux::YdotoolTyper::new(
        socket,
        false,
        config.ydotool_key_delay_ms,
        config.ydotool_key_hold_ms,
    );
    let notifier = typer::linux::LinuxNotifier::new();
    (Box::new(typer), Box::new(notifier))
}

#[cfg(target_os = "macos")]
fn build_platform_typer_notifier(
    _socket: Option<String>,
    _config: &Config,
) -> (Box<dyn typer::platform::Typer>, Box<dyn typer::platform::Notifier>) {
    let typer = typer::macos::MacOSTyper::new(false);
    let notifier = typer::macos::MacOSNotifier::new();
    (Box::new(typer), Box::new(notifier))
}
