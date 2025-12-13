//! Checkpoint Saver Daemon
//!
//! Watches for checkpoint files and handles:
//! - Automatic deletion of old checkpoints (keep N latest)
//! - Optional zstd compression
//! - Future: Cloud transfer (S3/GCS)
//!
//! Usage:
//!   cargo run --release -- --config config.toml
//!   cargo run --release -- --watch-dir checkpoints/phase8_10b_japanese

use anyhow::{Context, Result};
use clap::Parser;
use glob::glob;
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Checkpoint Saver Daemon
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to config file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Directory to watch (overrides config)
    #[arg(short, long)]
    watch_dir: Option<PathBuf>,

    /// Number of checkpoints to keep (overrides config)
    #[arg(short, long)]
    keep: Option<usize>,

    /// Enable compression (overrides config)
    #[arg(long)]
    compress: bool,

    /// Disable compression (overrides config)
    #[arg(long)]
    no_compress: bool,
}

#[derive(Debug, Deserialize, Clone)]
struct AppConfig {
    watcher: WatcherConfig,
    cleanup: CleanupConfig,
    compression: CompressionConfig,
}

#[derive(Debug, Deserialize, Clone)]
struct WatcherConfig {
    path: PathBuf,
    pattern: String,
}

#[derive(Debug, Deserialize, Clone)]
struct CleanupConfig {
    keep_latest: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct CompressionConfig {
    enabled: bool,
    level: i32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            watcher: WatcherConfig {
                path: PathBuf::from("checkpoints/phase8_10b_japanese"),
                pattern: "step_*.pt".to_string(),
            },
            cleanup: CleanupConfig { keep_latest: 2 },
            compression: CompressionConfig {
                enabled: true,
                level: 3,
            },
        }
    }
}

fn load_config(path: Option<&Path>) -> Result<AppConfig> {
    match path {
        Some(p) if p.exists() => {
            let content = fs::read_to_string(p)?;
            Ok(toml::from_str(&content)?)
        }
        _ => {
            info!("Using default config");
            Ok(AppConfig::default())
        }
    }
}

/// Get list of checkpoint files sorted by modification time (newest first)
fn get_checkpoints(dir: &Path, pattern: &str) -> Vec<PathBuf> {
    let glob_pattern = dir.join(pattern);
    let glob_str = glob_pattern.to_string_lossy();

    let mut files: Vec<_> = glob(&glob_str)
        .unwrap_or_else(|_| panic!("Invalid glob pattern: {}", glob_str))
        .filter_map(|e| e.ok())
        .filter(|p| !p.to_string_lossy().ends_with(".zst")) // Exclude compressed files
        .collect();

    // Sort by modification time (newest first)
    files.sort_by(|a, b| {
        let a_time = fs::metadata(a).and_then(|m| m.modified()).ok();
        let b_time = fs::metadata(b).and_then(|m| m.modified()).ok();
        b_time.cmp(&a_time)
    });

    files
}

/// Delete old checkpoints, keeping only the latest N
fn cleanup_old_checkpoints(dir: &Path, pattern: &str, keep: usize) -> Result<usize> {
    let checkpoints = get_checkpoints(dir, pattern);
    let mut deleted = 0;

    if checkpoints.len() > keep {
        for old_ckpt in checkpoints.iter().skip(keep) {
            match fs::remove_file(old_ckpt) {
                Ok(_) => {
                    info!("🗑️  Deleted old checkpoint: {}", old_ckpt.display());
                    deleted += 1;

                    // Also delete compressed version if exists
                    let compressed = PathBuf::from(format!("{}.zst", old_ckpt.display()));
                    if compressed.exists() {
                        let _ = fs::remove_file(&compressed);
                        info!("🗑️  Deleted compressed: {}", compressed.display());
                    }
                }
                Err(e) => {
                    warn!("Failed to delete {}: {}", old_ckpt.display(), e);
                }
            }
        }
    }

    Ok(deleted)
}

/// Compress a checkpoint file using zstd
fn compress_checkpoint(path: &Path, level: i32) -> Result<PathBuf> {
    let output_path = PathBuf::from(format!("{}.zst", path.display()));

    // Skip if already compressed
    if output_path.exists() {
        debug!("Already compressed: {}", output_path.display());
        return Ok(output_path);
    }

    info!("📦 Compressing: {} (level {})", path.display(), level);

    let mut input = File::open(path)?;
    let mut input_data = Vec::new();
    input.read_to_end(&mut input_data)?;

    let compressed = zstd::encode_all(&input_data[..], level)?;

    let mut output = File::create(&output_path)?;
    output.write_all(&compressed)?;

    let original_size = input_data.len() as f64 / 1024.0 / 1024.0;
    let compressed_size = compressed.len() as f64 / 1024.0 / 1024.0;
    let ratio = compressed_size / original_size * 100.0;

    info!(
        "✅ Compressed: {:.1} MB → {:.1} MB ({:.1}%)",
        original_size, compressed_size, ratio
    );

    Ok(output_path)
}

/// Process a new checkpoint file
fn process_checkpoint(path: &Path, config: &AppConfig) -> Result<()> {
    let filename = path.file_name().unwrap_or_default().to_string_lossy();

    // Skip non-checkpoint files
    if !filename.starts_with("step_") || !filename.ends_with(".pt") {
        return Ok(());
    }

    // Skip temporary files
    if filename.ends_with(".tmp") || filename.ends_with(".pt.zst") {
        return Ok(());
    }

    info!("📁 New checkpoint detected: {}", path.display());

    // Compress if enabled
    if config.compression.enabled {
        match compress_checkpoint(path, config.compression.level) {
            Ok(_) => {}
            Err(e) => warn!("Compression failed: {}", e),
        }
    }

    // Cleanup old checkpoints
    if let Some(parent) = path.parent() {
        match cleanup_old_checkpoints(parent, &config.watcher.pattern, config.cleanup.keep_latest) {
            Ok(n) if n > 0 => info!("Cleaned up {} old checkpoint(s)", n),
            Err(e) => warn!("Cleanup failed: {}", e),
            _ => {}
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "checkpoint_saver=info".to_string()),
        )
        .init();

    let args = Args::parse();

    // Load config
    let mut config = load_config(args.config.as_deref())?;

    // Override from CLI args
    if let Some(dir) = args.watch_dir {
        config.watcher.path = dir;
    }
    if let Some(keep) = args.keep {
        config.cleanup.keep_latest = keep;
    }
    if args.compress {
        config.compression.enabled = true;
    }
    if args.no_compress {
        config.compression.enabled = false;
    }

    info!("🚀 Checkpoint Saver Daemon");
    info!("   Watch dir: {}", config.watcher.path.display());
    info!("   Pattern: {}", config.watcher.pattern);
    info!("   Keep latest: {}", config.cleanup.keep_latest);
    info!(
        "   Compression: {} (level {})",
        if config.compression.enabled {
            "enabled"
        } else {
            "disabled"
        },
        config.compression.level
    );

    // Ensure watch directory exists
    if !config.watcher.path.exists() {
        fs::create_dir_all(&config.watcher.path)?;
        info!("Created watch directory: {}", config.watcher.path.display());
    }

    // Initial cleanup
    let deleted = cleanup_old_checkpoints(
        &config.watcher.path,
        &config.watcher.pattern,
        config.cleanup.keep_latest,
    )?;
    if deleted > 0 {
        info!("Initial cleanup: deleted {} old checkpoint(s)", deleted);
    }

    // Set up file watcher
    let (tx, rx) = channel();
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        },
        Config::default().with_poll_interval(Duration::from_secs(1)),
    )?;

    watcher.watch(&config.watcher.path, RecursiveMode::NonRecursive)?;

    info!("👀 Watching for checkpoints... (Ctrl+C to stop)");

    // Main event loop
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(event) => {
                if let notify::EventKind::Create(_) | notify::EventKind::Modify(_) = event.kind {
                    for path in event.paths {
                        if let Err(e) = process_checkpoint(&path, &config) {
                            error!("Error processing {}: {}", path.display(), e);
                        }
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Periodic check (in case events are missed)
                continue;
            }
            Err(e) => {
                error!("Watcher error: {}", e);
                break;
            }
        }
    }

    Ok(())
}
