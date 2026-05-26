use crate::config::Config;
use anyhow::Result;
use chrono::Utc;
use std::future::Future;
use std::path::PathBuf;
use std::time::Instant;
use tokio::task::JoinHandle;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

const STATUS_FLUSH_SECONDS: u64 = 5;
const BACKOFF_JITTER_RATIO: f64 = 0.25;

/// Policy passed to every component supervisor. Bundles backoff and circuit-breaker
/// thresholds so the call sites stay readable.
#[derive(Clone, Copy)]
struct SupervisorPolicy {
    initial_backoff_secs: u64,
    max_backoff_secs: u64,
    restart_budget: u32,
    restart_budget_window_secs: u64,
}

impl SupervisorPolicy {
    fn from_config(config: &Config) -> Self {
        let initial = config.reliability.channel_initial_backoff_secs.max(1);
        let max = config.reliability.channel_max_backoff_secs.max(initial);
        Self {
            initial_backoff_secs: initial,
            max_backoff_secs: max,
            restart_budget: config.reliability.component_restart_budget.max(1),
            restart_budget_window_secs: config
                .reliability
                .component_restart_budget_window_secs
                .max(1),
        }
    }
}

/// Apply ±BACKOFF_JITTER_RATIO jitter to a backoff value. Returns the effective
/// sleep duration in seconds.
fn jittered_backoff(secs: u64) -> Duration {
    let base = secs as f64;
    let factor = 1.0 + (rand::random::<f64>() * 2.0 - 1.0) * BACKOFF_JITTER_RATIO;
    let jittered = (base * factor).max(0.0);
    Duration::from_secs_f64(jittered)
}

pub async fn run(config: Config, host: String, port: u16) -> Result<()> {
    let policy = SupervisorPolicy::from_config(&config);
    let shutdown = CancellationToken::new();

    crate::health::mark_component_ok("daemon");

    if config.heartbeat.enabled {
        let _ =
            crate::heartbeat::engine::HeartbeatEngine::ensure_heartbeat_file(&config.workspace_dir)
                .await;
    }

    let mut handles: Vec<JoinHandle<()>> =
        vec![spawn_state_writer(config.clone(), shutdown.clone())];

    {
        let gateway_cfg = config.clone();
        let gateway_host = host.clone();
        handles.push(spawn_component_supervisor(
            "gateway",
            policy,
            shutdown.clone(),
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
                policy,
                shutdown.clone(),
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
        let heartbeat_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "heartbeat",
            policy,
            shutdown.clone(),
            move || {
                let cfg = heartbeat_cfg.clone();
                let cancel = heartbeat_shutdown.clone();
                async move { Box::pin(run_heartbeat_worker(cfg, cancel)).await }
            },
        ));
    }

    if config.cron.enabled {
        let scheduler_cfg = config.clone();
        handles.push(spawn_component_supervisor(
            "scheduler",
            policy,
            shutdown.clone(),
            move || {
                let cfg = scheduler_cfg.clone();
                async move { crate::cron::scheduler::run(cfg).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("scheduler");
        tracing::info!("Cron disabled; scheduler supervisor not started");
    }

    if config.autonomy.loop_enabled {
        let autonomy_cfg = config.clone();
        let autonomy_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "autonomy_loop",
            policy,
            shutdown.clone(),
            move || {
                let cfg = autonomy_cfg.clone();
                let cancel = autonomy_shutdown.clone();
                async move { Box::pin(run_autonomy_loop(cfg, cancel)).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("autonomy_loop");
        tracing::info!("Autonomy loop disabled (autonomy.loop_enabled = false)");
    }

    if config.taskqueue.enabled {
        let tq_cfg = config.clone();
        let tq_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "taskqueue",
            policy,
            shutdown.clone(),
            move || {
                let cfg = tq_cfg.clone();
                let cancel = tq_shutdown.clone();
                async move { Box::pin(run_taskqueue_worker(cfg, cancel)).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("taskqueue");
        tracing::info!("TaskQueue disabled (taskqueue.enabled = false)");
    }

    if config.skillforge.enabled {
        let sf_cfg = config.clone();
        let sf_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "skillforge",
            policy,
            shutdown.clone(),
            move || {
                let cfg = sf_cfg.clone();
                let cancel = sf_shutdown.clone();
                async move { Box::pin(run_skillforge_worker(cfg, cancel)).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("skillforge");
        tracing::info!("SkillForge disabled (skillforge.enabled = false)");
    }

    if config.life.enabled {
        let life_cfg = config.clone();
        let life_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "life",
            policy,
            shutdown.clone(),
            move || {
                let cfg = life_cfg.clone();
                let cancel = life_shutdown.clone();
                async move { Box::pin(run_life_supervisor(cfg, cancel)).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("life");
        tracing::info!("Life loop disabled (life.enabled = false)");
    }

    if config.sce.enabled {
        let sce_cfg = config.clone();
        let sce_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "sce",
            policy,
            shutdown.clone(),
            move || {
                let cfg = sce_cfg.clone();
                let cancel = sce_shutdown.clone();
                async move { Box::pin(run_sce_supervisor(cfg, cancel)).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("sce");
        tracing::info!("SCE disabled (sce.enabled = false)");
    }

    if config.consciousness.enabled {
        let con_cfg = config.clone();
        let con_shutdown = shutdown.clone();
        handles.push(spawn_component_supervisor(
            "consciousness",
            policy,
            shutdown.clone(),
            move || {
                let cfg = con_cfg.clone();
                let cancel = con_shutdown.clone();
                async move { Box::pin(run_consciousness_supervisor(cfg, cancel)).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("consciousness");
        tracing::info!("Consciousness disabled (consciousness.enabled = false)");
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
            policy,
            shutdown.clone(),
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
                policy,
                shutdown.clone(),
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
            let hb_shutdown = shutdown.clone();
            handles.push(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                let start = std::time::Instant::now();
                loop {
                    tokio::select! {
                        () = hb_shutdown.cancelled() => return,
                        _ = interval.tick() => {}
                    }
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
    tracing::info!("Shutdown signal received; cancelling supervised components");

    // Cancel cooperatively; give components a short grace period to exit.
    shutdown.cancel();
    let grace = Duration::from_secs(5);
    let drain = async {
        for handle in &mut handles {
            let _ = handle.await;
        }
    };
    if tokio::time::timeout(grace, drain).await.is_err() {
        tracing::warn!("Components did not exit within grace; aborting remaining tasks");
        for handle in &handles {
            handle.abort();
        }
        for handle in handles {
            let _ = handle.await;
        }
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

fn spawn_state_writer(config: Config, shutdown: CancellationToken) -> JoinHandle<()> {
    tokio::spawn(async move {
        let path = state_file_path(&config);
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        let mut interval = tokio::time::interval(Duration::from_secs(STATUS_FLUSH_SECONDS));
        loop {
            tokio::select! {
                () = shutdown.cancelled() => return,
                _ = interval.tick() => {}
            }
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

/// Spawn a supervisor task that runs `run_component` and restarts it on failure
/// with exponential backoff + jitter. Stops cleanly when `shutdown` is cancelled
/// or when the restart budget is exhausted (circuit breaker).
fn spawn_component_supervisor<F, Fut>(
    name: &'static str,
    policy: SupervisorPolicy,
    shutdown: CancellationToken,
    mut run_component: F,
) -> JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    tokio::spawn(async move {
        let mut backoff = policy.initial_backoff_secs.max(1);
        let max_backoff = policy.max_backoff_secs.max(backoff);
        let window = Duration::from_secs(policy.restart_budget_window_secs.max(1));
        let mut window_start: Option<Instant> = None;
        let mut restarts_in_window: u32 = 0;

        loop {
            if shutdown.is_cancelled() {
                tracing::info!("Supervisor '{name}' exiting (shutdown requested)");
                return;
            }

            crate::health::mark_component_ok(name);

            let run_result = tokio::select! {
                () = shutdown.cancelled() => {
                    tracing::info!("Supervisor '{name}' cancelled mid-run");
                    return;
                }
                result = run_component() => result,
            };

            match run_result {
                Ok(()) => {
                    crate::health::mark_component_error(name, "component exited unexpectedly");
                    tracing::warn!("Daemon component '{name}' exited unexpectedly");
                    // Clean exit — reset backoff since the component ran successfully
                    backoff = policy.initial_backoff_secs.max(1);
                }
                Err(e) => {
                    crate::health::mark_component_error(name, e.to_string());
                    tracing::error!("Daemon component '{name}' failed: {e}");
                }
            }

            crate::health::bump_component_restart(name);

            // Circuit-breaker accounting (sliding window).
            let now = Instant::now();
            match window_start {
                Some(start) if now.duration_since(start) <= window => {
                    restarts_in_window = restarts_in_window.saturating_add(1);
                }
                _ => {
                    window_start = Some(now);
                    restarts_in_window = 1;
                }
            }
            if restarts_in_window >= policy.restart_budget {
                let msg = format!(
                    "restart budget exceeded ({} restarts in {}s); supervisor giving up",
                    restarts_in_window,
                    window.as_secs(),
                );
                crate::health::mark_component_error(name, &msg);
                tracing::error!("Daemon component '{name}': {msg}");
                return;
            }

            // Backoff with jitter; cancel-aware sleep.
            let sleep_for = jittered_backoff(backoff);
            tokio::select! {
                () = shutdown.cancelled() => return,
                () = tokio::time::sleep(sleep_for) => {}
            }

            // Double backoff AFTER sleeping so first error uses initial_backoff
            backoff = backoff.saturating_mul(2).min(max_backoff);
        }
    })
}

async fn run_heartbeat_worker(config: Config, shutdown: CancellationToken) -> Result<()> {
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
        tokio::select! {
            () = shutdown.cancelled() => return Ok(()),
            _ = interval.tick() => {}
        }

        let tasks = engine.collect_tasks().await?;
        if tasks.is_empty() {
            continue;
        }

        for task in tasks {
            if shutdown.is_cancelled() {
                return Ok(());
            }
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

/// Append-only record of an autonomy decision. Persisted as JSONL so cognitive
/// layers and operators can audit what the loop did, and so we have crash-safe
/// outcome history independent of the goals DB.
#[derive(serde::Serialize)]
struct DecisionLogEntry<'a> {
    ts: String,
    goal_id: &'a str,
    title: &'a str,
    /// One of: "verified" | "agent_error" | "verification_failed" |
    /// "reverted_shutdown" | "budget_blocked" | "orphan_recovered" | "timed_out".
    outcome: &'a str,
    duration_ms: u128,
    evidence_preview: String,
}

/// Revert any `in_progress` goals to `approved`. The autonomy loop is the only
/// writer that sets `in_progress`, so any such row at startup (or after a long
/// gap) is by definition an orphan — left over from a daemon crash, kill, or
/// budget trip. This makes the autonomy loop self-healing across restarts.
fn recover_orphaned_goals(config: &Config, workspace: &std::path::Path) -> usize {
    let in_progress = match crate::goals::list(config, Some(crate::goals::GoalStatus::InProgress)) {
        Ok(list) => list,
        Err(e) => {
            tracing::warn!("Autonomy loop: orphan recovery failed to list: {e}");
            return 0;
        }
    };
    let mut recovered = 0_usize;
    for goal in in_progress {
        let reason = "recovered orphaned in_progress goal (daemon restart or interruption)";
        match crate::goals::revert_to_approved(config, &goal.id, reason) {
            Ok(_) => {
                tracing::info!(
                    "Autonomy loop: recovered orphaned goal '{}' ({})",
                    goal.id,
                    goal.title
                );
                append_decision_log(
                    workspace,
                    &DecisionLogEntry {
                        ts: chrono::Utc::now().to_rfc3339(),
                        goal_id: &goal.id,
                        title: &goal.title,
                        outcome: "orphan_recovered",
                        duration_ms: 0,
                        evidence_preview: reason.to_string(),
                    },
                );
                recovered += 1;
            }
            Err(e) => {
                tracing::warn!(
                    "Autonomy loop: failed to recover orphaned goal '{}': {e}",
                    goal.id
                );
            }
        }
    }
    recovered
}

/// Budget gate. Returns `Ok(None)` when dispatch is allowed; `Ok(Some(reason))`
/// when over budget; `Err(_)` if the tracker can't be read. The convention
/// `max_cost_per_day_cents == 0` means "unlimited" for backward compatibility.
fn check_daily_budget(
    tracker: Option<&crate::cost::CostTracker>,
    max_cents: u32,
) -> Result<Option<String>> {
    if max_cents == 0 {
        return Ok(None);
    }
    let Some(tracker) = tracker else {
        return Ok(None);
    };
    let today = chrono::Utc::now().date_naive();
    let spent_usd = tracker
        .get_daily_cost(today)
        .map_err(|e| anyhow::anyhow!("cost tracker error: {e}"))?;
    let budget_usd = f64::from(max_cents) / 100.0;
    if spent_usd >= budget_usd {
        Ok(Some(format!(
            "daily budget exceeded (${spent_usd:.2} of ${budget_usd:.2})"
        )))
    } else {
        Ok(None)
    }
}

fn append_decision_log(workspace: &std::path::Path, entry: &DecisionLogEntry<'_>) {
    let dir = workspace.join("autonomy");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("Autonomy loop: failed to create decision log dir: {e}");
        return;
    }
    let path = dir.join("decisions.jsonl");
    let mut line = match serde_json::to_string(entry) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("Autonomy loop: failed to serialize decision: {e}");
            return;
        }
    };
    line.push('\n');
    use std::io::Write as _;
    match std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                tracing::warn!("Autonomy loop: failed to append decision: {e}");
            }
        }
        Err(e) => tracing::warn!("Autonomy loop: failed to open decision log: {e}"),
    }
}

/// Propose `RecurringFailure` goals for any health component whose restart count
/// is at or above `threshold`. Dedupes by title against open goals so we don't
/// spam the queue. Returns the number of newly proposed goals.
fn propose_drift_goals(config: &Config, threshold: u32) -> usize {
    let snapshot = crate::health::snapshot_json();
    let Some(components) = snapshot.get("components").and_then(|v| v.as_object()) else {
        return 0;
    };

    let existing_titles: std::collections::HashSet<String> = [
        crate::goals::GoalStatus::Proposed,
        crate::goals::GoalStatus::Approved,
        crate::goals::GoalStatus::InProgress,
    ]
    .iter()
    .flat_map(|status| crate::goals::list(config, Some(*status)).unwrap_or_default())
    .map(|g| g.title)
    .collect();

    let mut proposed = 0_usize;
    for (name, value) in components {
        let restart_count = value
            .get("restart_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if u32::try_from(restart_count).unwrap_or(0) < threshold {
            continue;
        }
        let title = format!("Investigate recurring failures in '{name}'");
        if existing_titles.contains(&title) {
            continue;
        }
        let last_error = value
            .get("last_error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(no last error recorded)");
        let goal = crate::goals::Goal {
            id: uuid::Uuid::new_v4().to_string(),
            title: title.clone(),
            description: format!(
                "Component '{name}' has restarted {restart_count} times. Latest error: {last_error}. Diagnose the root cause and propose remediation."
            ),
            source: crate::goals::GoalSource::RecurringFailure,
            status: crate::goals::GoalStatus::Proposed,
            priority: 7,
            proposed_at: chrono::Utc::now(),
            approved_at: None,
            completed_at: None,
            evidence: None,
            success_criteria: vec![format!(
                "health.{name}.status returns to ok and stays stable"
            )],
            verification_method: crate::goals::VerificationMethod::HealthOk {
                component: name.clone(),
            },
        };
        match crate::goals::propose(config, &goal) {
            Ok(_) => {
                tracing::info!(
                    "Autonomy loop: proposed recurring-failure goal for '{name}' ({restart_count} restarts)"
                );
                proposed += 1;
            }
            Err(e) => {
                tracing::warn!("Autonomy loop: failed to propose drift goal for '{name}': {e}");
            }
        }
    }
    proposed
}

fn truncate_evidence(s: &str) -> String {
    const MAX: usize = 4000;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
}

/// Goal-driven autonomy loop. Polls approved goals and dispatches them through
/// `agent::run`. On verified success the goal is marked completed; on agent
/// failure or failed verification it reverts to approved so the next tick can
/// retry. When `self_trigger_enabled` is on, drift detection over the health
/// snapshot also proposes new `RecurringFailure` goals on a slower cadence.
async fn run_autonomy_loop(config: Config, shutdown: CancellationToken) -> Result<()> {
    let poll_secs = config.autonomy.loop_poll_secs.max(5);
    let mut interval = tokio::time::interval(Duration::from_secs(poll_secs));

    let self_trigger_enabled = config.autonomy.self_trigger_enabled;
    let self_trigger_interval =
        Duration::from_secs(config.autonomy.self_trigger_check_interval_secs.max(60));
    let self_trigger_threshold = config.autonomy.self_trigger_restart_threshold;
    let mut last_self_trigger = Instant::now();
    let budget_cents = config.autonomy.max_cost_per_day_cents;
    let goal_timeout = Duration::from_secs(config.autonomy.goal_timeout_secs.max(30));

    // Cost tracker shares the on-disk JSONL store with `agent::run`'s tracker.
    let cost_tracker =
        crate::cost::CostTracker::new(config.cost.clone(), &config.workspace_dir).ok();
    if budget_cents > 0 && cost_tracker.is_none() {
        tracing::warn!(
            "Autonomy loop: budget configured (${:.2}/day) but cost tracker init failed; budget gate disabled",
            f64::from(budget_cents) / 100.0
        );
    }

    // Sweep any in_progress rows left behind by a prior crash/shutdown.
    let orphans = recover_orphaned_goals(&config, &config.workspace_dir);
    if orphans > 0 {
        tracing::info!("Autonomy loop: startup recovered {orphans} orphaned goal(s)");
    }

    tracing::info!(
        "Autonomy loop starting (poll every {}s, self_trigger={}, budget_cents={})",
        poll_secs,
        self_trigger_enabled,
        budget_cents
    );

    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                tracing::info!("Autonomy loop shutting down");
                return Ok(());
            }
            _ = interval.tick() => {}
        }

        // #2B: budget gate — applied before drift proposals AND goal dispatch.
        let budget_block = match check_daily_budget(cost_tracker.as_ref(), budget_cents) {
            Ok(reason) => reason,
            Err(e) => {
                tracing::warn!("Autonomy loop: budget check failed: {e}");
                None
            }
        };
        if let Some(reason) = budget_block {
            crate::health::mark_component_error("autonomy_loop", &reason);
            tracing::warn!("Autonomy loop: skipping tick ({reason})");
            append_decision_log(
                &config.workspace_dir,
                &DecisionLogEntry {
                    ts: chrono::Utc::now().to_rfc3339(),
                    goal_id: "",
                    title: "(tick-skipped)",
                    outcome: "budget_blocked",
                    duration_ms: 0,
                    evidence_preview: reason,
                },
            );
            continue;
        }

        // #3: drift-driven proposal on a slower cadence than the goal poll.
        if self_trigger_enabled && last_self_trigger.elapsed() >= self_trigger_interval {
            let count = propose_drift_goals(&config, self_trigger_threshold);
            if count > 0 {
                tracing::info!("Autonomy loop: drift proposed {count} new goal(s)");
            }
            last_self_trigger = Instant::now();
        }

        let approved = match crate::goals::list(&config, Some(crate::goals::GoalStatus::Approved)) {
            Ok(list) => list,
            Err(e) => {
                tracing::warn!("Autonomy loop: failed to list goals: {e}");
                crate::health::mark_component_error("autonomy_loop", e.to_string());
                continue;
            }
        };

        let Some(goal) = approved.into_iter().next() else {
            crate::health::mark_component_ok("autonomy_loop");
            continue;
        };

        if let Err(e) = crate::goals::set_in_progress(&config, &goal.id) {
            tracing::warn!("Autonomy loop: failed to claim goal '{}': {e}", goal.id);
            continue;
        }

        let criteria_block = if goal.success_criteria.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nSuccess criteria (ALL must be satisfied):\n{}",
                goal.success_criteria
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  {}. {c}", i + 1))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        let prompt = format!(
            "[Autonomous Goal] (id={}, priority={}) {}\n\n{}{criteria_block}",
            goal.id, goal.priority, goal.title, goal.description
        );
        let temp = config.default_temperature;
        let start = Instant::now();

        tracing::info!(
            "Autonomy loop: dispatching goal '{}' ({})",
            goal.id,
            goal.title
        );

        // #2C: per-goal wall-clock timeout, racing against shutdown.
        let dispatch_result = tokio::select! {
            () = shutdown.cancelled() => {
                let _ = crate::goals::revert_to_approved(
                    &config,
                    &goal.id,
                    "shutdown during execution",
                );
                append_decision_log(
                    &config.workspace_dir,
                    &DecisionLogEntry {
                        ts: chrono::Utc::now().to_rfc3339(),
                        goal_id: &goal.id,
                        title: &goal.title,
                        outcome: "reverted_shutdown",
                        duration_ms: start.elapsed().as_millis(),
                        evidence_preview: String::new(),
                    },
                );
                return Ok(());
            }
            timed = tokio::time::timeout(
                goal_timeout,
                crate::agent::run(
                    config.clone(),
                    Some(prompt),
                    None,
                    None,
                    temp,
                    vec![],
                ),
            ) => timed,
        };

        let dispatch = match dispatch_result {
            Ok(result) => result,
            Err(_elapsed) => {
                let reason = format!(
                    "agent execution timed out after {}s",
                    goal_timeout.as_secs()
                );
                let _ = crate::goals::revert_to_approved(&config, &goal.id, &reason);
                crate::health::mark_component_error("autonomy_loop", &reason);
                tracing::warn!("Autonomy loop: goal '{}' {}", goal.id, reason);
                append_decision_log(
                    &config.workspace_dir,
                    &DecisionLogEntry {
                        ts: chrono::Utc::now().to_rfc3339(),
                        goal_id: &goal.id,
                        title: &goal.title,
                        outcome: "timed_out",
                        duration_ms: start.elapsed().as_millis(),
                        evidence_preview: reason,
                    },
                );
                continue;
            }
        };

        match dispatch {
            Ok(agent_response) => {
                // #5: verify before marking complete.
                let verification = crate::goals::verify_goal(&goal, &agent_response).await;
                let elapsed_ms = start.elapsed().as_millis();
                match verification {
                    Ok(outcome) if outcome.passed => {
                        let evidence = format!(
                            "VERIFIED ({})\n---\n{}",
                            outcome.evidence,
                            truncate_evidence(&agent_response)
                        );
                        if let Err(e) = crate::goals::complete(&config, &goal.id, &evidence) {
                            tracing::warn!("Autonomy loop: failed to mark goal complete: {e}");
                        } else {
                            crate::health::mark_component_ok("autonomy_loop");
                            tracing::info!(
                                "Autonomy loop: goal '{}' completed (verified)",
                                goal.id
                            );
                        }
                        append_decision_log(
                            &config.workspace_dir,
                            &DecisionLogEntry {
                                ts: chrono::Utc::now().to_rfc3339(),
                                goal_id: &goal.id,
                                title: &goal.title,
                                outcome: "verified",
                                duration_ms: elapsed_ms,
                                evidence_preview: truncate_evidence(&outcome.evidence),
                            },
                        );
                    }
                    Ok(outcome) => {
                        let reason = format!("verification failed: {}", outcome.evidence);
                        let _ = crate::goals::revert_to_approved(&config, &goal.id, &reason);
                        crate::health::mark_component_error("autonomy_loop", &reason);
                        tracing::warn!(
                            "Autonomy loop: goal '{}' verification failed: {}",
                            goal.id,
                            outcome.evidence
                        );
                        append_decision_log(
                            &config.workspace_dir,
                            &DecisionLogEntry {
                                ts: chrono::Utc::now().to_rfc3339(),
                                goal_id: &goal.id,
                                title: &goal.title,
                                outcome: "verification_failed",
                                duration_ms: elapsed_ms,
                                evidence_preview: truncate_evidence(&outcome.evidence),
                            },
                        );
                    }
                    Err(e) => {
                        let reason = format!("verification error: {e}");
                        let _ = crate::goals::revert_to_approved(&config, &goal.id, &reason);
                        crate::health::mark_component_error("autonomy_loop", &reason);
                        tracing::warn!("Autonomy loop: goal '{}' verifier crashed: {e}", goal.id);
                        append_decision_log(
                            &config.workspace_dir,
                            &DecisionLogEntry {
                                ts: chrono::Utc::now().to_rfc3339(),
                                goal_id: &goal.id,
                                title: &goal.title,
                                outcome: "verification_failed",
                                duration_ms: elapsed_ms,
                                evidence_preview: reason,
                            },
                        );
                    }
                }
            }
            Err(e) => {
                let reason = format!("agent execution failed: {e}");
                let _ = crate::goals::revert_to_approved(&config, &goal.id, &reason);
                crate::health::mark_component_error("autonomy_loop", &reason);
                tracing::warn!("Autonomy loop: goal '{}' failed: {e}", goal.id);
                append_decision_log(
                    &config.workspace_dir,
                    &DecisionLogEntry {
                        ts: chrono::Utc::now().to_rfc3339(),
                        goal_id: &goal.id,
                        title: &goal.title,
                        outcome: "agent_error",
                        duration_ms: start.elapsed().as_millis(),
                        evidence_preview: reason,
                    },
                );
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

async fn run_taskqueue_worker(config: Config, shutdown: CancellationToken) -> Result<()> {
    let poll = Duration::from_secs(config.taskqueue.poll_interval_secs.max(5));
    let mut interval = tokio::time::interval(poll);

    tracing::info!("TaskQueue worker starting (poll every {}s)", poll.as_secs());

    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                tracing::info!("TaskQueue worker shutting down");
                return Ok(());
            }
            _ = interval.tick() => {}
        }

        let dequeue_result: Result<Option<crate::taskqueue::TaskItem>> =
            crate::taskqueue::dequeue(&config);
        let task = match dequeue_result {
            Ok(Some(t)) => t,
            Ok(None) => {
                crate::health::mark_component_ok("taskqueue");
                continue;
            }
            Err(e) => {
                tracing::warn!("TaskQueue: dequeue error: {e}");
                crate::health::mark_component_error("taskqueue", e.to_string());
                continue;
            }
        };

        tracing::info!(
            "TaskQueue: dispatching task '{}' (priority={:?})",
            task.id,
            task.priority
        );
        let task_id = task.id.clone();
        let prompt = format!(
            "[TaskQueue Dispatch] (id={}, priority={:?}, source={:?})\n\n{}",
            task.id, task.priority, task.source, task.task
        );

        let start = Instant::now();
        match tokio::time::timeout(
            Duration::from_secs(600),
            crate::agent::run(
                config.clone(),
                Some(prompt),
                None,
                None,
                config.default_temperature,
                vec![],
            ),
        )
        .await
        {
            Ok(Ok(response)) => {
                let _ = crate::taskqueue::complete(&config, &task_id, &response);
                crate::health::mark_component_ok("taskqueue");
                tracing::info!(
                    "TaskQueue: task '{}' completed ({}ms)",
                    task_id,
                    start.elapsed().as_millis()
                );
            }
            Ok(Err(e)) => {
                let reason = format!("agent error: {e}");
                let _ = crate::taskqueue::fail(&config, &task_id, &reason);
                crate::health::mark_component_error("taskqueue", &reason);
                tracing::warn!("TaskQueue: task '{}' failed: {e}", task_id);
            }
            Err(_) => {
                let _ = crate::taskqueue::fail(&config, &task_id, "execution timed out (600s)");
                crate::health::mark_component_error("taskqueue", "task timed out");
                tracing::warn!("TaskQueue: task '{}' timed out", task_id);
            }
        }
    }
}

async fn run_skillforge_worker(config: Config, shutdown: CancellationToken) -> Result<()> {
    let interval_secs = config.skillforge.scan_interval_hours.max(1) * 3600;
    tracing::info!(
        "SkillForge worker starting (scan every {}h)",
        config.skillforge.scan_interval_hours
    );

    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                tracing::info!("SkillForge worker shutting down");
                return Ok(());
            }
            () = tokio::time::sleep(Duration::from_secs(interval_secs)) => {}
        }

        tracing::info!("SkillForge: starting scan");
        let sf_config = config.skillforge.clone();
        match crate::skillforge::SkillForge::new(sf_config).forge().await {
            Ok(report) => {
                crate::health::mark_component_ok("skillforge");
                tracing::info!(
                    "SkillForge: scan complete — discovered={}, integrated={}",
                    report.discovered,
                    report.auto_integrated
                );
            }
            Err(e) => {
                crate::health::mark_component_error("skillforge", e.to_string());
                tracing::warn!("SkillForge: scan failed: {e}");
            }
        }
    }
}

async fn run_life_supervisor(config: Config, shutdown: CancellationToken) -> Result<()> {
    use std::sync::Arc;

    let observer: Arc<dyn crate::observability::Observer> =
        Arc::from(crate::observability::create_observer(&config.observability));
    let mem: Arc<dyn crate::memory::traits::Memory> =
        Arc::from(crate::memory::create_memory_with_storage(
            &config.memory,
            Some(&config.storage.provider.config),
            &config.workspace_dir,
            config.api_key.as_deref(),
        )?);
    let provider_name = config.default_provider.as_deref().unwrap_or("openrouter");
    let provider: Arc<dyn crate::providers::traits::Provider> = Arc::from(
        crate::providers::create_provider(provider_name, config.api_key.as_deref())?,
    );
    let integration = Arc::new(tokio::sync::Mutex::new(
        crate::cosmic::IntegrationMeter::new(),
    ));

    let mut life = crate::life::LifeLoop::new(
        mem,
        provider,
        vec![],
        observer,
        integration,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        config.life.clone(),
    );

    tracing::info!(
        "Life loop starting (tick every {}s)",
        config.life.tick_interval_secs
    );

    tokio::select! {
        () = shutdown.cancelled() => {
            tracing::info!("Life loop shutting down");
            Ok(())
        }
        result = life.run() => {
            result
        }
    }
}

async fn run_sce_supervisor(config: Config, shutdown: CancellationToken) -> Result<()> {
    use std::sync::Arc;
    use tokio::sync::RwLock;

    let mem_rw: Arc<RwLock<dyn crate::memory::traits::Memory>> =
        Arc::new(RwLock::new(crate::memory::NoneMemory));

    let soul = crate::soul::model::SoulModel::default();
    let identity = crate::continuity::types::Identity {
        core: crate::continuity::types::IdentityCore {
            name: "ZeroClaw".into(),
            constitution_hash: String::new(),
            creation_epoch: 0,
            immutable_values: Vec::new(),
        },
        preferences: Vec::new(),
        narrative: Vec::new(),
        commitments: Vec::new(),
        session_count: 0,
    };
    let quantum_brain = crate::quantum::brain::QuantumBrainEngine::new();

    let mission = "autonomous self-continuity".to_string();
    let principles = vec!["integrity".into(), "coherence".into(), "growth".into()];

    let mut engine = crate::sce::SelfContinuityEngine::new(
        soul,
        identity,
        mission,
        principles,
        mem_rw,
        quantum_brain,
    );

    let tick_interval = Duration::from_secs(config.sce.tick_interval_secs.max(10));
    let mut interval = tokio::time::interval(tick_interval);

    tracing::info!("SCE starting (tick every {}s)", tick_interval.as_secs());

    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                tracing::info!("SCE shutting down");
                return Ok(());
            }
            _ = interval.tick() => {}
        }

        match engine.tick() {
            Ok(result) => {
                crate::health::mark_component_ok("sce");
                tracing::debug!(
                    "SCE tick {} — proposals={}, actions={}, coherence={:.3}",
                    result.tick,
                    result.proposals_generated,
                    result.actions_taken,
                    result.coherence
                );
            }
            Err(e) => {
                crate::health::mark_component_error("sce", e.to_string());
                tracing::warn!("SCE tick error: {e}");
            }
        }
    }
}

async fn run_consciousness_supervisor(config: Config, shutdown: CancellationToken) -> Result<()> {
    use crate::consciousness::orchestrator::ConsciousnessConfig as CConfig;

    let cc = CConfig {
        debate_rounds: config.consciousness.debate_rounds,
        approval_threshold: config.consciousness.approval_threshold,
        bus_capacity: config.consciousness.bus_capacity,
        max_discourse_depth: config.consciousness.max_discourse_depth,
        ..CConfig::default()
    };

    let mut orch = crate::consciousness::ConsciousnessOrchestrator::new(cc);

    let tick_interval = Duration::from_secs(5);
    let mut interval = tokio::time::interval(tick_interval);

    tracing::info!("Consciousness supervisor starting (tick every 5s)");

    loop {
        tokio::select! {
            () = shutdown.cancelled() => {
                tracing::info!("Consciousness supervisor shutting down");
                return Ok(());
            }
            _ = interval.tick() => {}
        }

        let result = orch.tick();
        crate::health::mark_component_ok("consciousness");
        tracing::debug!(
            "Consciousness tick — coherence={:.3}, proposals={}",
            result.coherence,
            result.proposals_generated
        );
    }
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

    fn test_policy() -> SupervisorPolicy {
        SupervisorPolicy {
            initial_backoff_secs: 1,
            max_backoff_secs: 1,
            restart_budget: u32::MAX,
            restart_budget_window_secs: 3600,
        }
    }

    #[tokio::test]
    async fn supervisor_marks_error_and_restart_on_failure() {
        let token = CancellationToken::new();
        let handle = spawn_component_supervisor(
            "daemon-test-fail",
            test_policy(),
            token.clone(),
            || async { anyhow::bail!("boom") },
        );

        tokio::time::sleep(Duration::from_millis(50)).await;
        token.cancel();
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
        let token = CancellationToken::new();
        let handle = spawn_component_supervisor(
            "daemon-test-exit",
            test_policy(),
            token.clone(),
            || async { Ok(()) },
        );

        tokio::time::sleep(Duration::from_millis(50)).await;
        token.cancel();
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

    #[tokio::test]
    async fn supervisor_honors_cancellation() {
        let token = CancellationToken::new();
        let handle = spawn_component_supervisor(
            "daemon-test-cancel",
            test_policy(),
            token.clone(),
            || async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(())
            },
        );

        // Cancel before grace period; supervisor must return cleanly without abort().
        tokio::time::sleep(Duration::from_millis(50)).await;
        token.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
        assert!(result.is_ok(), "supervisor did not exit on cancellation");
    }

    #[tokio::test]
    async fn supervisor_circuit_breaks_after_restart_budget() {
        let token = CancellationToken::new();
        let policy = SupervisorPolicy {
            initial_backoff_secs: 1,
            max_backoff_secs: 1,
            restart_budget: 2,
            restart_budget_window_secs: 3600,
        };
        let handle =
            spawn_component_supervisor("daemon-test-circuit", policy, token.clone(), || async {
                anyhow::bail!("repeated failure")
            });

        // Budget=2 with backoff=1s and jitter ±25% should trip within ~3s.
        let result = tokio::time::timeout(Duration::from_secs(6), handle).await;
        assert!(
            result.is_ok(),
            "supervisor did not exit after exhausting restart budget"
        );

        let snapshot = crate::health::snapshot_json();
        let component = &snapshot["components"]["daemon-test-circuit"];
        assert_eq!(component["status"], "error");
        assert!(component["last_error"]
            .as_str()
            .unwrap_or("")
            .contains("restart budget exceeded"));
    }

    #[test]
    fn jitter_stays_within_bounds() {
        for _ in 0..100 {
            let d = jittered_backoff(10);
            let secs = d.as_secs_f64();
            assert!(
                (7.49..=12.51).contains(&secs),
                "jittered backoff out of bounds: {secs}"
            );
        }
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
    fn append_decision_log_writes_jsonl() {
        let tmp = TempDir::new().unwrap();
        let entry = DecisionLogEntry {
            ts: "2026-01-01T00:00:00Z".to_string(),
            goal_id: "g1",
            title: "test goal",
            outcome: "verified",
            duration_ms: 42,
            evidence_preview: "ok".to_string(),
        };
        append_decision_log(tmp.path(), &entry);

        let path = tmp.path().join("autonomy").join("decisions.jsonl");
        let contents = std::fs::read_to_string(&path).expect("log file should exist");
        assert!(contents.contains("\"goal_id\":\"g1\""));
        assert!(contents.contains("\"outcome\":\"verified\""));
        assert!(contents.ends_with('\n'));

        // Append a second entry and verify both lines are present.
        let entry2 = DecisionLogEntry {
            ts: "2026-01-01T00:01:00Z".to_string(),
            goal_id: "g2",
            title: "next",
            outcome: "agent_error",
            duration_ms: 10,
            evidence_preview: "boom".to_string(),
        };
        append_decision_log(tmp.path(), &entry2);
        let contents2 = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents2.lines().count(), 2);
        assert!(contents2.contains("\"goal_id\":\"g2\""));
    }

    #[test]
    fn recover_orphaned_goals_reverts_in_progress() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        // Propose two goals, approve them, then claim them (set_in_progress).
        for n in 0..2 {
            let id = format!("orphan-{n}");
            let goal = crate::goals::Goal {
                id: id.clone(),
                title: format!("orphan {n}"),
                description: String::new(),
                source: crate::goals::GoalSource::default(),
                status: crate::goals::GoalStatus::Proposed,
                priority: 5,
                proposed_at: chrono::Utc::now(),
                approved_at: None,
                completed_at: None,
                evidence: None,
                success_criteria: Vec::new(),
                verification_method: crate::goals::VerificationMethod::default(),
            };
            crate::goals::propose(&config, &goal).unwrap();
            crate::goals::approve(&config, &id).unwrap();
            crate::goals::set_in_progress(&config, &id).unwrap();
        }

        // Simulate a crash — call recover at "startup".
        let recovered = recover_orphaned_goals(&config, &config.workspace_dir);
        assert_eq!(recovered, 2, "both in_progress goals must be recovered");

        // Now they must be back to approved and pickable.
        let approved =
            crate::goals::list(&config, Some(crate::goals::GoalStatus::Approved)).unwrap();
        assert!(approved.iter().any(|g| g.id == "orphan-0"));
        assert!(approved.iter().any(|g| g.id == "orphan-1"));

        // Decision log must have one entry per recovery.
        let log_path = config
            .workspace_dir
            .join("autonomy")
            .join("decisions.jsonl");
        let log_contents = std::fs::read_to_string(&log_path).unwrap();
        assert_eq!(log_contents.lines().count(), 2);
        assert!(log_contents.contains("\"outcome\":\"orphan_recovered\""));
    }

    #[test]
    fn recover_orphaned_goals_only_touches_in_progress() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        // One goal in each status: proposed, approved, in_progress, completed, rejected.
        let mk = |id: &str| crate::goals::Goal {
            id: id.to_string(),
            title: id.to_string(),
            description: String::new(),
            source: crate::goals::GoalSource::default(),
            status: crate::goals::GoalStatus::Proposed,
            priority: 5,
            proposed_at: chrono::Utc::now(),
            approved_at: None,
            completed_at: None,
            evidence: None,
            success_criteria: Vec::new(),
            verification_method: crate::goals::VerificationMethod::default(),
        };
        crate::goals::propose(&config, &mk("a-proposed")).unwrap();
        crate::goals::propose(&config, &mk("b-approved")).unwrap();
        crate::goals::approve(&config, "b-approved").unwrap();
        crate::goals::propose(&config, &mk("c-in-progress")).unwrap();
        crate::goals::approve(&config, "c-in-progress").unwrap();
        crate::goals::set_in_progress(&config, "c-in-progress").unwrap();
        crate::goals::propose(&config, &mk("d-completed")).unwrap();
        crate::goals::approve(&config, "d-completed").unwrap();
        crate::goals::set_in_progress(&config, "d-completed").unwrap();
        crate::goals::complete(&config, "d-completed", "done").unwrap();
        crate::goals::propose(&config, &mk("e-rejected")).unwrap();
        crate::goals::reject(&config, "e-rejected", "no").unwrap();

        let recovered = recover_orphaned_goals(&config, &config.workspace_dir);
        assert_eq!(recovered, 1, "exactly one in_progress should be recovered");

        // The in_progress goal should now be approved; everything else unchanged.
        assert_eq!(
            crate::goals::get(&config, "a-proposed").unwrap().status,
            crate::goals::GoalStatus::Proposed
        );
        assert_eq!(
            crate::goals::get(&config, "b-approved").unwrap().status,
            crate::goals::GoalStatus::Approved
        );
        assert_eq!(
            crate::goals::get(&config, "c-in-progress").unwrap().status,
            crate::goals::GoalStatus::Approved,
            "orphan c must be reverted to approved"
        );
        assert_eq!(
            crate::goals::get(&config, "d-completed").unwrap().status,
            crate::goals::GoalStatus::Completed,
            "completed goals must be left alone"
        );
        assert_eq!(
            crate::goals::get(&config, "e-rejected").unwrap().status,
            crate::goals::GoalStatus::Rejected,
            "rejected goals must be left alone"
        );
    }

    #[test]
    fn check_daily_budget_zero_is_unlimited() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);
        let tracker =
            crate::cost::CostTracker::new(config.cost.clone(), &config.workspace_dir).unwrap();
        let result = check_daily_budget(Some(&tracker), 0).unwrap();
        assert!(result.is_none(), "zero budget means unlimited");
    }

    #[test]
    fn check_daily_budget_no_tracker_is_noop() {
        let result = check_daily_budget(None, 5000).unwrap();
        assert!(result.is_none(), "absent tracker disables enforcement");
    }

    #[test]
    fn check_daily_budget_blocks_when_over() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp);
        // Default CostConfig is disabled; enable for this test so record_usage persists.
        config.cost.enabled = true;
        let tracker =
            crate::cost::CostTracker::new(config.cost.clone(), &config.workspace_dir).unwrap();

        // Inject one usage record large enough to bust a 1¢ budget.
        let usage = crate::cost::TokenUsage {
            model: "test-model".into(),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cost_usd: 0.50,
            timestamp: chrono::Utc::now(),
        };
        tracker.record_usage(usage).unwrap();

        // 1¢ budget should be exceeded by $0.50 spend.
        let blocked = check_daily_budget(Some(&tracker), 1).unwrap();
        assert!(
            blocked.is_some(),
            "expected budget block at $0.50 spent vs $0.01 budget"
        );
        let msg = blocked.unwrap();
        assert!(msg.contains("budget exceeded"), "unexpected msg: {msg}");

        // A generous $100 budget should pass.
        let allowed = check_daily_budget(Some(&tracker), 10_000).unwrap();
        assert!(allowed.is_none());
    }

    #[test]
    fn propose_drift_goals_creates_recurring_failure_goal() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        // Mark a component as failed enough times to cross the threshold.
        let unique = format!("drift-test-{}", uuid::Uuid::new_v4());
        for _ in 0..6 {
            crate::health::mark_component_error(&unique, "synthetic failure");
            crate::health::bump_component_restart(&unique);
        }

        let proposed = propose_drift_goals(&config, 3);
        assert!(proposed >= 1, "expected at least one drift-proposed goal");

        let pending =
            crate::goals::list(&config, Some(crate::goals::GoalStatus::Proposed)).unwrap();
        let title = format!("Investigate recurring failures in '{unique}'");
        let matching: Vec<_> = pending.iter().filter(|g| g.title == title).collect();
        assert_eq!(matching.len(), 1, "exactly one matching goal expected");
        assert_eq!(
            matching[0].source,
            crate::goals::GoalSource::RecurringFailure
        );
        // Re-running should not duplicate: the title already exists.
        let again = propose_drift_goals(&config, 3);
        assert_eq!(again, 0, "drift goals must dedupe by title");
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
