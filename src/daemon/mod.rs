use crate::config::Config;
use anyhow::Result;
use chrono::Utc;
use std::future::Future;
use std::path::PathBuf;
use tokio::task::JoinHandle;
use tokio::time::Duration;

const STATUS_FLUSH_SECONDS: u64 = 5;

pub async fn run(config: Config, host: String, port: u16) -> Result<()> {
    let initial_backoff = config.reliability.channel_initial_backoff_secs.max(1);
    let max_backoff = config
        .reliability
        .channel_max_backoff_secs
        .max(initial_backoff);

    crate::health::mark_component_ok("daemon");

    if config.heartbeat.enabled {
        let _ =
            crate::heartbeat::engine::HeartbeatEngine::ensure_heartbeat_file(&config.workspace_dir)
                .await;
    }

    let mut handles: Vec<JoinHandle<()>> = vec![spawn_state_writer(config.clone())];

    {
        let gateway_cfg = config.clone();
        let gateway_host = host.clone();
        handles.push(spawn_component_supervisor(
            "gateway",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = gateway_cfg.clone();
                let host = gateway_host.clone();
                async move { crate::gateway::run_gateway(&host, port, cfg).await }
            },
        ));
    }

    {
        if has_supervised_channels(&config) {
            let channels_cfg = config.clone();
            handles.push(spawn_component_supervisor(
                "channels",
                initial_backoff,
                max_backoff,
                move || {
                    let cfg = channels_cfg.clone();
                    async move { crate::channels::start_channels(cfg).await }
                },
            ));
        } else {
            crate::health::mark_component_ok("channels");
            tracing::info!("No real-time channels configured; channel supervisor disabled");
        }
    }

    if config.heartbeat.enabled {
        let heartbeat_cfg = config.clone();
        handles.push(spawn_component_supervisor(
            "heartbeat",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = heartbeat_cfg.clone();
                async move { Box::pin(run_heartbeat_worker(cfg)).await }
            },
        ));
    }

    if config.cron.enabled {
        let scheduler_cfg = config.clone();
        handles.push(spawn_component_supervisor(
            "scheduler",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = scheduler_cfg.clone();
                async move { crate::cron::scheduler::run(cfg).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("scheduler");
        tracing::info!("Cron disabled; scheduler supervisor not started");
    }

    let bot_ids: Vec<String> = config.bots.keys().cloned().collect();
    for bot_id in &bot_ids {
        let resolved = config.resolve_bot_config(bot_id);
        let bot_workspace = resolved.workspace_dir.clone();
        if let Err(e) = std::fs::create_dir_all(&bot_workspace) {
            tracing::error!("Failed to create workspace for bot '{bot_id}': {e}");
            continue;
        }

        let bot_port = resolved.gateway.port;
        let bot_name: &'static str = Box::leak(format!("gateway-{bot_id}").into_boxed_str());
        let gw_cfg = resolved.clone();
        let gw_host = host.clone();
        handles.push(spawn_component_supervisor(
            bot_name,
            initial_backoff,
            max_backoff,
            move || {
                let cfg = gw_cfg.clone();
                let h = gw_host.clone();
                async move { crate::gateway::run_gateway(&h, bot_port, cfg).await }
            },
        ));

        if has_supervised_channels(&resolved) {
            let ch_name: &'static str = Box::leak(format!("channels-{bot_id}").into_boxed_str());
            let ch_cfg = resolved.clone();
            handles.push(spawn_component_supervisor(
                ch_name,
                initial_backoff,
                max_backoff,
                move || {
                    let cfg = ch_cfg.clone();
                    async move { crate::channels::start_channels(cfg).await }
                },
            ));
        }

        let store_workspace = config.workspace_dir.clone();
        let reg_bot_id = bot_id.clone();
        let reg_name = config
            .bots
            .get(bot_id)
            .and_then(|b| b.name.clone())
            .unwrap_or_else(|| bot_id.clone());
        let reg_host = host.clone();
        let reg_port = resolved.gateway.port;
        let reg_channels: Vec<&str> = {
            let cc = &resolved.channels_config;
            let mut ch = Vec::new();
            if cc.telegram.is_some() {
                ch.push("telegram");
            }
            if cc.discord.is_some() {
                ch.push("discord");
            }
            if cc.slack.is_some() {
                ch.push("slack");
            }
            if cc.whatsapp.is_some() {
                ch.push("whatsapp");
            }
            if cc.matrix.is_some() {
                ch.push("matrix");
            }
            if cc.signal.is_some() {
                ch.push("signal");
            }
            if cc.email.is_some() {
                ch.push("email");
            }
            if cc.irc.is_some() {
                ch.push("irc");
            }
            if cc.lark.is_some() {
                ch.push("lark");
            }
            if cc.dingtalk.is_some() {
                ch.push("dingtalk");
            }
            ch
        };
        let channels_json = serde_json::to_string(&reg_channels).unwrap_or_else(|_| "[]".into());

        if let Ok(store) = crate::control::store::ControlStore::open(&store_workspace) {
            let now = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let bot_entry = crate::control::store::Bot {
                id: reg_bot_id.clone(),
                name: reg_name.clone(),
                host: reg_host.clone(),
                port: reg_port,
                status: "online".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                last_heartbeat: now.clone(),
                channels: channels_json,
                provider: resolved.default_provider.clone().unwrap_or_default(),
                memory_backend: resolved.memory.backend.to_string(),
                uptime_secs: 0,
                registered_at: now,
                workspace_dir: bot_workspace.display().to_string(),
            };
            let _ = store.upsert_bot(&bot_entry);
        }

        {
            let hb_store_workspace = config.workspace_dir.clone();
            let hb_bot_id = bot_id.clone();
            let hb_host = host.clone();
            let hb_port = resolved.gateway.port;
            handles.push(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                let start = std::time::Instant::now();
                loop {
                    interval.tick().await;
                    if let Ok(store) =
                        crate::control::store::ControlStore::open(&hb_store_workspace)
                    {
                        let now = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
                        let uptime = start.elapsed().as_secs() as i64;
                        let bot_update = crate::control::store::Bot {
                            id: hb_bot_id.clone(),
                            name: hb_bot_id.clone(),
                            host: hb_host.clone(),
                            port: hb_port,
                            status: "online".to_string(),
                            version: env!("CARGO_PKG_VERSION").to_string(),
                            last_heartbeat: now.clone(),
                            channels: "[]".to_string(),
                            provider: String::new(),
                            memory_backend: String::new(),
                            uptime_secs: uptime,
                            registered_at: now,
                            workspace_dir: String::new(),
                        };
                        let _ = store.upsert_bot(&bot_update);
                    }
                }
            }));
        }

        println!("   Bot '{bot_id}': http://{host}:{}", resolved.gateway.port);
    }

    println!("🧠 ZeroClaw daemon started");
    println!("   Gateway:  http://{host}:{port}");
    println!("   Components: gateway, channels, heartbeat, scheduler");
    if !bot_ids.is_empty() {
        println!("   Bots: {} isolated workspace(s)", bot_ids.len());
    }
    println!("   Ctrl+C to stop");

    tokio::signal::ctrl_c().await?;
    crate::health::mark_component_error("daemon", "shutdown requested");

    for handle in &handles {
        handle.abort();
    }
    for handle in handles {
        let _ = handle.await;
    }

    Ok(())
}

pub fn state_file_path(config: &Config) -> PathBuf {
    config
        .config_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join("daemon_state.json")
}

fn spawn_state_writer(config: Config) -> JoinHandle<()> {
    tokio::spawn(async move {
        let path = state_file_path(&config);
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        let mut interval = tokio::time::interval(Duration::from_secs(STATUS_FLUSH_SECONDS));
        loop {
            interval.tick().await;
            let mut json = crate::health::snapshot_json();
            if let Some(obj) = json.as_object_mut() {
                obj.insert(
                    "written_at".into(),
                    serde_json::json!(Utc::now().to_rfc3339()),
                );
            }
            let data = serde_json::to_vec_pretty(&json).unwrap_or_else(|_| b"{}".to_vec());
            let _ = tokio::fs::write(&path, data).await;
        }
    })
}

fn spawn_component_supervisor<F, Fut>(
    name: &'static str,
    initial_backoff_secs: u64,
    max_backoff_secs: u64,
    mut run_component: F,
) -> JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    tokio::spawn(async move {
        let mut backoff = initial_backoff_secs.max(1);
        let max_backoff = max_backoff_secs.max(backoff);

        loop {
            crate::health::mark_component_ok(name);
            match run_component().await {
                Ok(()) => {
                    crate::health::mark_component_error(name, "component exited unexpectedly");
                    tracing::warn!("Daemon component '{name}' exited unexpectedly");
                    // Clean exit — reset backoff since the component ran successfully
                    backoff = initial_backoff_secs.max(1);
                }
                Err(e) => {
                    crate::health::mark_component_error(name, e.to_string());
                    tracing::error!("Daemon component '{name}' failed: {e}");
                }
            }

            crate::health::bump_component_restart(name);
            tokio::time::sleep(Duration::from_secs(backoff)).await;
            // Double backoff AFTER sleeping so first error uses initial_backoff
            backoff = backoff.saturating_mul(2).min(max_backoff);
        }
    })
}

async fn run_heartbeat_worker(config: Config) -> Result<()> {
    let observer: std::sync::Arc<dyn crate::observability::Observer> =
        std::sync::Arc::from(crate::observability::create_observer(&config.observability));
    let engine = crate::heartbeat::engine::HeartbeatEngine::new(
        config.heartbeat.clone(),
        config.workspace_dir.clone(),
        observer,
    );

    let interval_mins = config.heartbeat.interval_minutes.max(5);
    let mut interval = tokio::time::interval(Duration::from_secs(u64::from(interval_mins) * 60));

    loop {
        interval.tick().await;

        let tasks = engine.collect_tasks().await?;
        if tasks.is_empty() {
            continue;
        }

        for task in tasks {
            let prompt = format!("[Heartbeat Task] {task}");
            let temp = config.default_temperature;
            if let Err(e) =
                crate::agent::run(config.clone(), Some(prompt), None, None, temp, vec![]).await
            {
                crate::health::mark_component_error("heartbeat", e.to_string());
                tracing::warn!("Heartbeat task failed: {e}");
            } else {
                crate::health::mark_component_ok("heartbeat");
            }
        }
    }
}

fn has_supervised_channels(config: &Config) -> bool {
    config.channels_config.telegram.is_some()
        || config.channels_config.discord.is_some()
        || config.channels_config.slack.is_some()
        || config.channels_config.imessage.is_some()
        || config.channels_config.matrix.is_some()
        || config.channels_config.signal.is_some()
        || config.channels_config.whatsapp.is_some()
        || config.channels_config.email.is_some()
        || config.channels_config.irc.is_some()
        || config.channels_config.lark.is_some()
        || config.channels_config.dingtalk.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(tmp: &TempDir) -> Config {
        let config = Config {
            workspace_dir: tmp.path().join("workspace"),
            config_path: tmp.path().join("config.toml"),
            ..Config::default()
        };
        std::fs::create_dir_all(&config.workspace_dir).unwrap();
        config
    }

    #[test]
    fn state_file_path_uses_config_directory() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        let path = state_file_path(&config);
        assert_eq!(path, tmp.path().join("daemon_state.json"));
    }

    #[tokio::test]
    async fn supervisor_marks_error_and_restart_on_failure() {
        let handle = spawn_component_supervisor("daemon-test-fail", 1, 1, || async {
            anyhow::bail!("boom")
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.abort();
        let _ = handle.await;

        let snapshot = crate::health::snapshot_json();
        let component = &snapshot["components"]["daemon-test-fail"];
        assert_eq!(component["status"], "error");
        assert!(component["restart_count"].as_u64().unwrap_or(0) >= 1);
        assert!(component["last_error"]
            .as_str()
            .unwrap_or("")
            .contains("boom"));
    }

    #[tokio::test]
    async fn supervisor_marks_unexpected_exit_as_error() {
        let handle = spawn_component_supervisor("daemon-test-exit", 1, 1, || async { Ok(()) });

        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.abort();
        let _ = handle.await;

        let snapshot = crate::health::snapshot_json();
        let component = &snapshot["components"]["daemon-test-exit"];
        assert_eq!(component["status"], "error");
        assert!(component["restart_count"].as_u64().unwrap_or(0) >= 1);
        assert!(component["last_error"]
            .as_str()
            .unwrap_or("")
            .contains("component exited unexpectedly"));
    }

    #[test]
    fn detects_no_supervised_channels() {
        let config = Config::default();
        assert!(!has_supervised_channels(&config));
    }

    #[test]
    fn detects_supervised_channels_present() {
        let mut config = Config::default();
        config.channels_config.telegram = Some(crate::config::TelegramConfig {
            bot_token: "token".into(),
            allowed_users: vec![],
            stream_mode: crate::config::StreamMode::default(),
            draft_update_interval_ms: 1000,
            mention_only: false,
            voice: crate::config::VoiceConfig::default(),
        });
        assert!(has_supervised_channels(&config));
    }

    #[test]
    fn detects_dingtalk_as_supervised_channel() {
        let mut config = Config::default();
        config.channels_config.dingtalk = Some(crate::config::schema::DingTalkConfig {
            client_id: "client_id".into(),
            client_secret: "client_secret".into(),
            allowed_users: vec!["*".into()],
        });
        assert!(has_supervised_channels(&config));
    }
}
