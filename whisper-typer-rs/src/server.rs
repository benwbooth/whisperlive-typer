use std::path::Path;
use std::process::Command;
use std::time::Duration;

use tokio::time::sleep;
use tracing::{error, info};

use crate::config::Config;

/// Check if the WhisperLive server is accepting WebSocket connections.
async fn is_server_running(host: &str, port: u16) -> bool {
    let uri = format!("ws://{host}:{port}");
    let timeout = Duration::from_secs(5);
    match tokio::time::timeout(timeout, tokio_tungstenite::connect_async(&uri)).await {
        Ok(Ok((mut ws, _))) => {
            let _ = ws.close(None).await;
            true
        }
        _ => false,
    }
}

/// Start the server via docker compose.
fn start_server_docker(server_dir: &str) -> bool {
    let compose = Path::new(server_dir).join("docker-compose.yml");
    if !compose.exists() {
        error!("docker-compose.yml not found in {server_dir}");
        return false;
    }

    info!("Starting server from {server_dir}...");
    match Command::new("docker")
        .args(["compose", "up", "-d"])
        .current_dir(server_dir)
        .output()
    {
        Ok(output) if output.status.success() => {
            info!("Docker compose started");
            true
        }
        Ok(output) => {
            error!(
                "Failed to start server: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            false
        }
        Err(e) => {
            error!("docker command not found: {e}");
            false
        }
    }
}

/// Start the server via launchctl (macOS).
#[cfg(target_os = "macos")]
fn start_server_macos() -> bool {
    info!("Starting MLX server via launchctl...");
    let result = Command::new("launchctl")
        .args(["start", "com.whispertyper.server"])
        .output();
    match result {
        Ok(output) if output.status.success() => {
            info!("MLX server started via launchctl");
            true
        }
        Ok(output) => {
            error!(
                "Failed to start server: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            false
        }
        Err(e) => {
            error!("launchctl error: {e}");
            false
        }
    }
}

fn start_server(server_dir: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        return start_server_macos();
    }
    #[cfg(not(target_os = "macos"))]
    {
        let dir = if server_dir.is_empty() {
            // Use the directory of the executable
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_string_lossy().into_owned()))
                .unwrap_or_else(|| ".".into())
        } else {
            server_dir.to_string()
        };
        start_server_docker(&dir)
    }
}

/// Wait for the server to become ready.
async fn wait_for_server(host: &str, port: u16, timeout_secs: u64) -> bool {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    while start.elapsed() < timeout {
        if is_server_running(host, port).await {
            info!("Server is ready");
            return true;
        }
        let remaining = timeout.saturating_sub(start.elapsed()).as_secs();
        info!("Waiting for server... ({remaining}s remaining)");
        sleep(Duration::from_secs(2)).await;
    }
    error!("Server did not start within {timeout_secs} seconds");
    false
}

/// Ensure the server is running, starting it if necessary.
pub async fn ensure_server_running(config: &Config) -> bool {
    if is_server_running(&config.host, config.port).await {
        info!("Server already running at {}:{}", config.host, config.port);
        return true;
    }

    if !config.auto_start_server {
        error!(
            "Server not running at {}:{} and auto_start is disabled",
            config.host, config.port
        );
        return false;
    }

    info!(
        "Server not running at {}:{}, starting...",
        config.host, config.port
    );

    let server_dir = config.server_dir.clone();
    let started = tokio::task::spawn_blocking(move || start_server(&server_dir))
        .await
        .unwrap_or(false);

    if !started {
        return false;
    }

    wait_for_server(&config.host, config.port, config.server_start_timeout).await
}
