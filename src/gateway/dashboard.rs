use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Json},
};
use serde_json::{json, Value};

use super::AppState;

fn check_auth(headers: &HeaderMap, state: &AppState) -> Option<(StatusCode, Json<Value>)> {
    if !state.pairing.require_pairing() {
        return None;
    }
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token = auth.strip_prefix("Bearer ").unwrap_or("");
    if !state.pairing.is_authenticated(token) {
        return Some((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Unauthorized — pair first via POST /pair"})),
        ));
    }
    None
}

pub async fn handle_dashboard() -> impl IntoResponse {
    Html(DASHBOARD_HTML)
}

pub async fn handle_api_status(State(state): State<AppState>) -> impl IntoResponse {
    let config = state.config.lock();

    let channels: Vec<&str> = {
        let mut ch = Vec::new();
        if config.channels_config.telegram.is_some() {
            ch.push("telegram");
        }
        if config.channels_config.discord.is_some() {
            ch.push("discord");
        }
        if config.channels_config.slack.is_some() {
            ch.push("slack");
        }
        if config.channels_config.mattermost.is_some() {
            ch.push("mattermost");
        }
        if config.channels_config.matrix.is_some() {
            ch.push("matrix");
        }
        if config.channels_config.whatsapp.is_some() {
            ch.push("whatsapp");
        }
        if config.channels_config.webhook.is_some() {
            ch.push("webhook");
        }
        if config.channels_config.signal.is_some() {
            ch.push("signal");
        }
        if config.channels_config.email.is_some() {
            ch.push("email");
        }
        if config.channels_config.irc.is_some() {
            ch.push("irc");
        }
        if config.channels_config.imessage.is_some() {
            ch.push("imessage");
        }
        if config.channels_config.lark.is_some() {
            ch.push("lark");
        }
        if config.channels_config.dingtalk.is_some() {
            ch.push("dingtalk");
        }
        if config.channels_config.qq.is_some() {
            ch.push("qq");
        }
        ch
    };

    let tools_enabled: Vec<String> = config.autonomy.auto_approve.clone();

    let agent_names: Vec<String> = config.agents.keys().cloned().collect();

    Json(json!({
        "provider": config.default_provider,
        "model": config.default_model,
        "temperature": config.default_temperature,
        "memory_backend": format!("{}", config.memory.backend),
        "channels": channels,
        "channels_count": channels.len(),
        "tools_enabled": tools_enabled,
        "tools_count": tools_enabled.len(),
        "agents": agent_names,
        "agents_count": agent_names.len(),
        "gateway": {
            "host": &config.gateway.host,
            "port": config.gateway.port,
            "require_pairing": config.gateway.require_pairing,
        },
        "security": {
            "autonomy_level": format!("{:?}", config.autonomy.level),
            "sandbox_enabled": config.security.sandbox.enabled,
        },
        "identity": {
            "format": &config.identity.format,
            "aieos_path": &config.identity.aieos_path,
        },
    }))
}

pub async fn handle_api_channels(State(state): State<AppState>) -> impl IntoResponse {
    let config = state.config.lock();
    let cc = &config.channels_config;

    let channels = json!([
        {
            "name": "telegram",
            "label": "Telegram",
            "category": "messaging",
            "enabled": cc.telegram.is_some(),
            "required_keys": ["bot_token", "allowed_users"],
            "optional_keys": ["stream_mode", "mention_only", "voice"],
            "hint": "Create a bot via @BotFather, get the token, add allowed user IDs."
        },
        {
            "name": "discord",
            "label": "Discord",
            "category": "messaging",
            "enabled": cc.discord.is_some(),
            "required_keys": ["bot_token"],
            "optional_keys": ["guild_id", "allowed_users", "listen_to_bots", "mention_only"],
            "hint": "Create a bot at discord.com/developers, enable Message Content intent."
        },
        {
            "name": "slack",
            "label": "Slack",
            "category": "messaging",
            "enabled": cc.slack.is_some(),
            "required_keys": ["bot_token"],
            "optional_keys": ["app_token", "channel_id", "allowed_users"],
            "hint": "Create a Slack app, add Bot Token Scopes, install to workspace."
        },
        {
            "name": "mattermost",
            "label": "Mattermost",
            "category": "messaging",
            "enabled": cc.mattermost.is_some(),
            "required_keys": ["url", "bot_token"],
            "optional_keys": ["channel_id", "allowed_users", "thread_replies", "mention_only"],
            "hint": "Create a bot account in Mattermost, use the Personal Access Token."
        },
        {
            "name": "matrix",
            "label": "Matrix",
            "category": "messaging",
            "enabled": cc.matrix.is_some(),
            "required_keys": ["homeserver", "access_token", "room_id", "allowed_users"],
            "optional_keys": ["user_id", "device_id"],
            "hint": "Register a bot user, generate access token, invite to room."
        },
        {
            "name": "whatsapp",
            "label": "WhatsApp",
            "category": "messaging",
            "enabled": cc.whatsapp.is_some(),
            "required_keys": ["access_token", "phone_number_id", "verify_token"],
            "optional_keys": ["app_secret", "allowed_numbers"],
            "hint": "Set up Meta Business API, configure webhook URL to /whatsapp."
        },
        {
            "name": "signal",
            "label": "Signal",
            "category": "messaging",
            "enabled": cc.signal.is_some(),
            "required_keys": ["http_url", "account"],
            "optional_keys": ["group_id", "allowed_from", "ignore_attachments", "ignore_stories"],
            "hint": "Run signal-cli REST API daemon, register a phone number."
        },
        {
            "name": "email",
            "label": "Email",
            "category": "communication",
            "enabled": cc.email.is_some(),
            "required_keys": ["imap_host", "smtp_host", "username", "password"],
            "optional_keys": ["imap_port", "smtp_port", "allowed_senders", "folder"],
            "hint": "Configure IMAP for receiving and SMTP for sending email."
        },
        {
            "name": "irc",
            "label": "IRC",
            "category": "communication",
            "enabled": cc.irc.is_some(),
            "required_keys": ["server", "nickname"],
            "optional_keys": ["port", "channels", "allowed_users", "server_password", "nickserv_password", "sasl_password", "verify_tls"],
            "hint": "Connect to any IRC server. Default port 6697 (TLS)."
        },
        {
            "name": "webhook",
            "label": "Webhook",
            "category": "integration",
            "enabled": cc.webhook.is_some(),
            "required_keys": ["port"],
            "optional_keys": ["secret"],
            "hint": "Generic HTTP webhook. POST JSON to receive, configurable secret."
        },
        {
            "name": "imessage",
            "label": "iMessage",
            "category": "messaging",
            "enabled": cc.imessage.is_some(),
            "required_keys": ["allowed_contacts"],
            "optional_keys": [],
            "hint": "macOS only. Uses AppleScript to read/send iMessages."
        },
        {
            "name": "lark",
            "label": "Lark / Feishu",
            "category": "enterprise",
            "enabled": cc.lark.is_some(),
            "required_keys": ["app_id", "app_secret"],
            "optional_keys": ["encrypt_key", "verification_token", "allowed_users", "use_feishu", "receive_mode", "port"],
            "hint": "Create app in Lark/Feishu console. Supports WebSocket (default) or webhook."
        },
        {
            "name": "dingtalk",
            "label": "DingTalk",
            "category": "enterprise",
            "enabled": cc.dingtalk.is_some(),
            "required_keys": ["client_id", "client_secret"],
            "optional_keys": ["allowed_users"],
            "hint": "Create a DingTalk enterprise bot, get AppKey and AppSecret."
        },
        {
            "name": "qq",
            "label": "QQ",
            "category": "messaging",
            "enabled": cc.qq.is_some(),
            "required_keys": ["app_id", "app_secret"],
            "optional_keys": ["allowed_users"],
            "hint": "Register at Tencent QQ Bot developer console."
        }
    ]);

    let enabled_count = [
        cc.telegram.is_some(),
        cc.discord.is_some(),
        cc.slack.is_some(),
        cc.mattermost.is_some(),
        cc.matrix.is_some(),
        cc.whatsapp.is_some(),
        cc.signal.is_some(),
        cc.email.is_some(),
        cc.irc.is_some(),
        cc.webhook.is_some(),
        cc.imessage.is_some(),
        cc.lark.is_some(),
        cc.dingtalk.is_some(),
        cc.qq.is_some(),
    ]
    .iter()
    .filter(|&&e| e)
    .count();

    Json(json!({
        "channels": channels,
        "total": 14,
        "enabled": enabled_count,
        "cli_enabled": cc.cli,
    }))
}

pub async fn handle_api_system(State(state): State<AppState>) -> impl IntoResponse {
    let config = state.config.lock();

    let has_api_key = |key: &str| -> bool { std::env::var(key).is_ok() };

    let providers = json!([
        { "name": "anthropic", "label": "Anthropic", "category": "frontier", "enabled": has_api_key("ANTHROPIC_API_KEY") || has_api_key("ANTHROPIC_OAUTH_TOKEN"), "env_var": "ANTHROPIC_API_KEY", "hint": "Claude models. Get key at console.anthropic.com" },
        { "name": "openai", "label": "OpenAI", "category": "frontier", "enabled": has_api_key("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY", "hint": "GPT-4/o models. Get key at platform.openai.com" },
        { "name": "openrouter", "label": "OpenRouter", "category": "aggregator", "enabled": has_api_key("OPENROUTER_API_KEY"), "env_var": "OPENROUTER_API_KEY", "hint": "Multi-model gateway. Access 100+ models via openrouter.ai" },
        { "name": "ollama", "label": "Ollama", "category": "local", "enabled": has_api_key("OLLAMA_API_KEY") || config.default_provider.as_deref() == Some("ollama"), "env_var": "OLLAMA_API_KEY", "hint": "Local models. Run ollama serve, pull models with ollama pull" },
        { "name": "gemini", "label": "Google Gemini", "category": "frontier", "enabled": has_api_key("GOOGLE_API_KEY") || has_api_key("GEMINI_API_KEY"), "env_var": "GOOGLE_API_KEY", "hint": "Gemini models. Get key at aistudio.google.com" },
        { "name": "groq", "label": "Groq", "category": "inference", "enabled": has_api_key("GROQ_API_KEY"), "env_var": "GROQ_API_KEY", "hint": "Ultra-fast inference. Get key at console.groq.com" },
        { "name": "mistral", "label": "Mistral", "category": "frontier", "enabled": has_api_key("MISTRAL_API_KEY"), "env_var": "MISTRAL_API_KEY", "hint": "Mistral/Mixtral models. Get key at console.mistral.ai" },
        { "name": "xai", "label": "xAI (Grok)", "category": "frontier", "enabled": has_api_key("XAI_API_KEY"), "env_var": "XAI_API_KEY", "hint": "Grok models. Get key at console.x.ai" },
        { "name": "deepseek", "label": "DeepSeek", "category": "frontier", "enabled": has_api_key("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY", "hint": "DeepSeek V3/R1 models. Get key at platform.deepseek.com" },
        { "name": "together", "label": "Together AI", "category": "inference", "enabled": has_api_key("TOGETHER_API_KEY"), "env_var": "TOGETHER_API_KEY", "hint": "Open-source model hosting. Get key at api.together.ai" },
        { "name": "fireworks", "label": "Fireworks AI", "category": "inference", "enabled": has_api_key("FIREWORKS_API_KEY"), "env_var": "FIREWORKS_API_KEY", "hint": "Fast open-source inference. Get key at fireworks.ai" },
        { "name": "perplexity", "label": "Perplexity", "category": "search", "enabled": has_api_key("PERPLEXITY_API_KEY"), "env_var": "PERPLEXITY_API_KEY", "hint": "Search-augmented models. Get key at perplexity.ai" },
        { "name": "cohere", "label": "Cohere", "category": "frontier", "enabled": has_api_key("COHERE_API_KEY"), "env_var": "COHERE_API_KEY", "hint": "Command-R models. Get key at dashboard.cohere.com" },
        { "name": "copilot", "label": "GitHub Copilot", "category": "aggregator", "enabled": has_api_key("GITHUB_TOKEN"), "env_var": "GITHUB_TOKEN", "hint": "Use Copilot API with GitHub token" },
        { "name": "minimax", "label": "MiniMax", "category": "china", "enabled": has_api_key("MINIMAX_API_KEY"), "env_var": "MINIMAX_API_KEY", "hint": "MiniMax models (international and CN)" },
        { "name": "glm", "label": "GLM / Zhipu", "category": "china", "enabled": has_api_key("GLM_API_KEY") || has_api_key("ZHIPU_API_KEY"), "env_var": "GLM_API_KEY", "hint": "GLM-4 models from Zhipu AI" },
        { "name": "moonshot", "label": "Moonshot / Kimi", "category": "china", "enabled": has_api_key("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY", "hint": "Kimi models from Moonshot AI" },
        { "name": "qwen", "label": "Qwen", "category": "china", "enabled": has_api_key("QWEN_API_KEY") || has_api_key("DASHSCOPE_API_KEY"), "env_var": "QWEN_API_KEY", "hint": "Qwen models from Alibaba (CN/Intl/US)" },
        { "name": "zai", "label": "ZAI / 01.AI", "category": "china", "enabled": has_api_key("ZAI_API_KEY"), "env_var": "ZAI_API_KEY", "hint": "Yi models from 01.AI" },
        { "name": "qianfan", "label": "Qianfan / Baidu", "category": "china", "enabled": has_api_key("QIANFAN_API_KEY"), "env_var": "QIANFAN_API_KEY", "hint": "ERNIE models from Baidu" },
        { "name": "codex", "label": "OpenAI Codex", "category": "inference", "enabled": has_api_key("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY", "hint": "Codex/code models via OpenAI API" }
    ]);

    let tools = json!([
        { "name": "shell", "category": "system", "hint": "Execute shell commands" },
        { "name": "file_read", "category": "system", "hint": "Read file contents" },
        { "name": "file_write", "category": "system", "hint": "Write/create files" },
        { "name": "memory_store", "category": "memory", "hint": "Store key-value in memory" },
        { "name": "memory_recall", "category": "memory", "hint": "Recall from memory by query" },
        { "name": "memory_forget", "category": "memory", "hint": "Delete a memory entry" },
        { "name": "browser", "category": "browser", "hint": "Browse and extract web content" },
        { "name": "browser_open", "category": "browser", "hint": "Open URL in browser" },
        { "name": "screenshot", "category": "browser", "hint": "Take browser screenshot" },
        { "name": "http_request", "category": "network", "hint": "Make HTTP requests" },
        { "name": "web_search", "category": "network", "hint": "Search the web (DDG/Brave)" },
        { "name": "git_operations", "category": "system", "hint": "Git commands (status, diff, log)" },
        { "name": "schedule", "category": "scheduling", "hint": "Schedule one-time tasks" },
        { "name": "cron_add", "category": "scheduling", "hint": "Add recurring cron job" },
        { "name": "cron_list", "category": "scheduling", "hint": "List all cron jobs" },
        { "name": "cron_remove", "category": "scheduling", "hint": "Remove a cron job" },
        { "name": "cron_run", "category": "scheduling", "hint": "Manually trigger a cron job" },
        { "name": "cron_runs", "category": "scheduling", "hint": "View cron run history" },
        { "name": "cron_update", "category": "scheduling", "hint": "Update an existing cron job" },
        { "name": "wallet_info", "category": "wallet", "hint": "Get wallet address and info" },
        { "name": "wallet_balance", "category": "wallet", "hint": "Check wallet balance" },
        { "name": "wallet_send", "category": "wallet", "hint": "Send native token" },
        { "name": "wallet_pay", "category": "wallet", "hint": "Pay to address or ENS" },
        { "name": "wallet_sign", "category": "wallet", "hint": "Sign a message" },
        { "name": "wallet_token_balance", "category": "wallet", "hint": "Check ERC-20 token balance" },
        { "name": "wallet_token_send", "category": "wallet", "hint": "Send ERC-20 tokens" },
        { "name": "hardware_board_info", "category": "hardware", "hint": "Get connected board info" },
        { "name": "hardware_memory_read", "category": "hardware", "hint": "Read memory from MCU via probe" },
        { "name": "hardware_memory_map", "category": "hardware", "hint": "Show MCU memory map" },
        { "name": "delegate", "category": "agent", "hint": "Delegate task to sub-agent" },
        { "name": "composio", "category": "integration", "hint": "Run Composio actions" },
        { "name": "pushover", "category": "integration", "hint": "Send push notifications" },
        { "name": "proxy_config", "category": "system", "hint": "View/update proxy settings" },
        { "name": "soul_status", "category": "soul", "hint": "Get soul/identity status" },
        { "name": "soul_reflect", "category": "soul", "hint": "Trigger self-reflection" },
        { "name": "soul_replicate", "category": "soul", "hint": "Replicate soul to another instance" },
        { "name": "image_info", "category": "media", "hint": "Analyze image metadata" }
    ]);

    let memory_backend = config.memory.backend.clone();
    let memory_backends = json!([
        { "name": "sqlite", "label": "SQLite", "enabled": memory_backend == "sqlite", "hint": "Default. Local file-based, zero setup." },
        { "name": "lucid", "label": "Lucid", "enabled": memory_backend == "lucid", "hint": "High-performance embedded engine with vector search." },
        { "name": "postgres", "label": "PostgreSQL", "enabled": memory_backend == "postgres", "hint": "Production-grade. Requires DATABASE_URL env var." },
        { "name": "markdown", "label": "Markdown", "enabled": memory_backend == "markdown", "hint": "File-based markdown storage. Human-readable." },
        { "name": "none", "label": "None", "enabled": memory_backend == "none", "hint": "No persistent memory. Stateless operation." }
    ]);

    let observer_type = config.observability.backend.clone();
    let observers = json!([
        { "name": "noop", "label": "Noop", "enabled": observer_type == "noop" || observer_type == "none", "hint": "No observability. Silent operation." },
        { "name": "log", "label": "Log", "enabled": observer_type == "log", "hint": "Structured logging to stdout/file." },
        { "name": "prometheus", "label": "Prometheus", "enabled": observer_type == "prometheus", "hint": "Exposes /metrics endpoint for scraping." },
        { "name": "otel", "label": "OpenTelemetry", "enabled": observer_type == "otel" || observer_type == "opentelemetry", "hint": "OTLP export. Set OTEL_EXPORTER_OTLP_ENDPOINT." },
        { "name": "verbose", "label": "Verbose", "enabled": observer_type == "verbose", "hint": "Detailed debug output for development." },
        { "name": "selfhealth", "label": "Self-Health", "enabled": observer_type == "selfhealth", "hint": "Internal health monitoring and alerts." }
    ]);

    let runtime_type = config.runtime.kind.clone();
    let runtimes = json!([
        { "name": "native", "label": "Native", "enabled": runtime_type == "native", "hint": "Direct host execution. Default and fastest." },
        { "name": "docker", "label": "Docker", "enabled": runtime_type == "docker", "hint": "Container isolation. Requires Docker daemon." },
        { "name": "wasm", "label": "WebAssembly", "enabled": runtime_type == "wasm", "hint": "Sandboxed WASM runtime. Experimental." }
    ]);

    let autonomy_level = format!("{:?}", config.autonomy.level);
    let security = json!({
        "autonomy_level": &autonomy_level,
        "workspace_only": config.autonomy.workspace_only,
        "sandbox_enabled": config.security.sandbox.enabled,
        "auto_approve": &config.autonomy.auto_approve,
        "levels": [
            { "name": "ReadOnly", "label": "Read-Only", "active": autonomy_level == "ReadOnly", "hint": "Agent can only read. No writes, no commands." },
            { "name": "Supervised", "label": "Supervised", "active": autonomy_level == "Supervised", "hint": "Default. Agent asks before risky actions." },
            { "name": "Full", "label": "Full Autonomy", "active": autonomy_level == "Full", "hint": "Agent acts without confirmation. Use with caution." }
        ]
    });

    let tunnel_provider = &config.tunnel.provider;
    let tunnels = json!([
        { "name": "none", "label": "None", "enabled": tunnel_provider == "none", "hint": "No tunnel. Direct access only." },
        { "name": "cloudflare", "label": "Cloudflare Tunnel", "enabled": tunnel_provider == "cloudflare" || config.tunnel.cloudflare.is_some(), "required_keys": ["token"], "hint": "Zero Trust tunnel. Get token from CF dashboard." },
        { "name": "tailscale", "label": "Tailscale", "enabled": tunnel_provider == "tailscale" || config.tunnel.tailscale.is_some(), "required_keys": [], "optional_keys": ["funnel", "hostname"], "hint": "Mesh VPN. Optional Funnel for public access." },
        { "name": "ngrok", "label": "ngrok", "enabled": tunnel_provider == "ngrok" || config.tunnel.ngrok.is_some(), "required_keys": ["auth_token"], "optional_keys": ["domain"], "hint": "Instant public URLs. Get token at ngrok.com" },
        { "name": "custom", "label": "Custom", "enabled": tunnel_provider == "custom" || config.tunnel.custom.is_some(), "required_keys": ["start_command"], "hint": "Any tunnel via command template. Use {port} placeholder." }
    ]);

    let active_provider = &config.default_provider;
    let active_model = &config.default_model;

    Json(json!({
        "providers": { "items": providers, "active": active_provider, "active_model": active_model },
        "channels": { "total": 14, "items": "see /api/channels" },
        "tools": { "items": tools, "total": 37 },
        "memory": { "items": memory_backends, "active": memory_backend },
        "observers": { "items": observers, "active": observer_type },
        "runtimes": { "items": runtimes, "active": runtime_type },
        "security": security,
        "tunnels": { "items": tunnels, "active": tunnel_provider },
    }))
}

pub async fn handle_api_config(State(state): State<AppState>) -> impl IntoResponse {
    let config = state.config.lock();

    let mut config_json = serde_json::to_value(&*config).unwrap_or(json!({}));

    if let Some(obj) = config_json.as_object_mut() {
        obj.remove("api_key");
        if let Some(channels) = obj.get_mut("channels_config") {
            if let Some(ch_obj) = channels.as_object_mut() {
                for (_name, ch_config) in ch_obj.iter_mut() {
                    if let Some(ch) = ch_config.as_object_mut() {
                        for field in SECRET_FIELDS {
                            ch.remove(*field);
                        }
                    }
                }
            }
        }
    }

    Json(config_json)
}

pub async fn handle_api_memories(State(state): State<AppState>) -> impl IntoResponse {
    match state.mem.list(None, None).await {
        Ok(entries) => {
            let items: Vec<serde_json::Value> = entries
                .iter()
                .map(|e| {
                    json!({
                        "key": e.key,
                        "content": if e.content.len() > 200 {
                            format!("{}...", &e.content[..200])
                        } else {
                            e.content.clone()
                        },
                        "category": format!("{:?}", e.category),
                        "timestamp": &e.timestamp,
                    })
                })
                .collect();
            (
                StatusCode::OK,
                Json(json!({ "count": items.len(), "entries": items })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn handle_api_metrics(State(state): State<AppState>) -> impl IntoResponse {
    if let Some(prom) = state
        .observer
        .as_any()
        .downcast_ref::<crate::observability::PrometheusObserver>()
    {
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain")],
            prom.encode(),
        )
            .into_response()
    } else {
        (
            StatusCode::OK,
            Json(json!({ "message": "Prometheus observer not active" })),
        )
            .into_response()
    }
}

pub async fn handle_admin_provider(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    if let Some(provider) = body.get("provider").and_then(|v| v.as_str()) {
        if !VALID_PROVIDERS.contains(&provider) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown provider: {provider}")})),
            );
        }
        config.default_provider = Some(provider.to_string());
    }
    if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
        config.default_model = Some(model.to_string());
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

pub async fn handle_admin_channel(
    headers: HeaderMap,
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    let cc = &mut config.channels_config;
    let result = match name.as_str() {
        "cli" => {
            cc.cli = body
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Ok(false)
        }
        "telegram" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.telegram = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid telegram config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "discord" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.discord = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid discord config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "slack" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.slack = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid slack config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "whatsapp" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.whatsapp = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid whatsapp config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "webhook" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.webhook = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid webhook config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "matrix" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.matrix = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid matrix config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "mattermost" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.mattermost = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid mattermost config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "signal" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.signal = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid signal config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "email" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.email = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid email config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "irc" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.irc = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid irc config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "imessage" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.imessage = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid imessage config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        "lark" => {
            if let Some(obj) = body.as_object() {
                let val = serde_json::to_value(obj).unwrap_or_default();
                match serde_json::from_value(val) {
                    Ok(c) => {
                        cc.lark = Some(c);
                        Ok(true)
                    }
                    Err(e) => Err(format!("Invalid lark config: {e}")),
                }
            } else {
                Err("Expected JSON object".into())
            }
        }
        other => Err(format!("Unknown channel: {other}")),
    };
    match result {
        Ok(restart) => match config.save() {
            Ok(()) => (
                StatusCode::OK,
                Json(json!({"ok": true, "restart_required": restart})),
            ),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Failed to save: {e}")})),
            ),
        },
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))),
    }
}

pub async fn handle_admin_channel_delete(
    headers: HeaderMap,
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    let cc = &mut config.channels_config;
    match name.as_str() {
        "telegram" => cc.telegram = None,
        "discord" => cc.discord = None,
        "slack" => cc.slack = None,
        "whatsapp" => cc.whatsapp = None,
        "webhook" => cc.webhook = None,
        "matrix" => cc.matrix = None,
        "mattermost" => cc.mattermost = None,
        "signal" => cc.signal = None,
        "email" => cc.email = None,
        "irc" => cc.irc = None,
        "imessage" => cc.imessage = None,
        "lark" => cc.lark = None,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown channel: {other}")})),
            );
        }
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

pub async fn handle_admin_memory(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    if let Some(backend) = body.get("backend").and_then(|v| v.as_str()) {
        if !VALID_MEMORY_BACKENDS.contains(&backend) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown memory backend: {backend}")})),
            );
        }
        config.memory.backend = backend.to_string();
    }
    if let Some(auto_save) = body.get("auto_save").and_then(|v| v.as_bool()) {
        config.memory.auto_save = auto_save;
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

pub async fn handle_admin_observer(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    if let Some(backend) = body.get("backend").and_then(|v| v.as_str()) {
        if !VALID_OBSERVER_BACKENDS.contains(&backend) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown observer backend: {backend}")})),
            );
        }
        config.observability.backend = backend.to_string();
    }
    if let Some(ep) = body.get("otel_endpoint").and_then(|v| v.as_str()) {
        config.observability.otel_endpoint = Some(ep.to_string());
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

pub async fn handle_admin_runtime(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    if let Some(kind) = body.get("kind").and_then(|v| v.as_str()) {
        if !VALID_RUNTIME_KINDS.contains(&kind) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown runtime kind: {kind}")})),
            );
        }
        config.runtime.kind = kind.to_string();
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

pub async fn handle_admin_security(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    if let Some(level) = body.get("level").and_then(|v| v.as_str()) {
        config.autonomy.level = match level {
            "ReadOnly" | "readonly" => crate::security::AutonomyLevel::ReadOnly,
            "Supervised" | "supervised" => crate::security::AutonomyLevel::Supervised,
            "Full" | "full" => crate::security::AutonomyLevel::Full,
            other => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!("Unknown autonomy level: {other}")})),
                );
            }
        };
    }
    if let Some(ws) = body.get("workspace_only").and_then(|v| v.as_bool()) {
        config.autonomy.workspace_only = ws;
    }
    if let Some(arr) = body.get("auto_approve").and_then(|v| v.as_array()) {
        config.autonomy.auto_approve = arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

pub async fn handle_admin_tunnel(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&headers, &state) {
        return err;
    }
    let mut config = state.config.lock();
    if let Some(provider) = body.get("provider").and_then(|v| v.as_str()) {
        if !VALID_TUNNEL_PROVIDERS.contains(&provider) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown tunnel provider: {provider}")})),
            );
        }
        config.tunnel.provider = provider.to_string();
    }
    match config.save() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "restart_required": true})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to save: {e}")})),
        ),
    }
}

const DASHBOARD_HTML: &str = r##"
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZeroClaw Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Sora:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
--bg:#0a0f1a;--surface:#111827;--surface-hover:#1a2332;--surface-muted:#1e293b;
--border:#1e3a5f;--border-light:#2d4a6f;
--text:#e2e8f0;--text-muted:#94a3b8;--text-dim:#64748b;
--accent:#3b82f6;--accent-hover:#2563eb;--accent-muted:rgba(59,130,246,.15);
--success:#22c55e;--success-muted:rgba(34,197,94,.15);
--warning:#eab308;--warning-muted:rgba(234,179,8,.15);
--danger:#ef4444;--danger-muted:rgba(239,68,68,.15);
--purple:#a855f7;--purple-muted:rgba(168,85,247,.15);
--sidebar-w:260px;--header-h:0px;
}
html,body{height:100%;overflow:hidden}
body{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}
h1,h2,h3,h4,h5,h6{font-family:'Sora',sans-serif;font-weight:600}
code,.mono{font-family:'JetBrains Mono',monospace;font-size:0.85em}
a{color:var(--accent);text-decoration:none}
@keyframes fadeInUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.fade-in{animation:fadeInUp .3s ease both}
.shimmer{background:linear-gradient(90deg,var(--surface) 25%,var(--surface-hover) 50%,var(--surface) 75%);background-size:200% 100%;animation:shimmer 1.5s infinite}
.sidebar{position:fixed;top:0;left:0;width:var(--sidebar-w);height:100vh;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;z-index:100;transition:transform .3s ease}
.sidebar-header{padding:20px 20px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px}
.sidebar-header svg{flex-shrink:0}
.sidebar-header h1{font-size:16px;font-weight:700;letter-spacing:-.02em}
.sidebar-header span{font-size:10px;color:var(--text-dim);display:block;margin-top:2px;font-family:'JetBrains Mono',monospace}
.nav-scroll{flex:1;overflow-y:auto;padding:12px 0}
.nav-scroll::-webkit-scrollbar{width:4px}
.nav-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.nav-group{margin-bottom:4px}
.nav-group-label{padding:8px 20px 4px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text-dim);cursor:pointer;display:flex;align-items:center;justify-content:space-between;user-select:none}
.nav-group-label:hover{color:var(--text-muted)}
.nav-group-label .arrow{transition:transform .2s;font-size:8px}
.nav-group.collapsed .arrow{transform:rotate(-90deg)}
.nav-group.collapsed .nav-items{display:none}
.nav-items{padding:2px 0}
.nav-item{display:flex;align-items:center;gap:10px;padding:8px 20px;cursor:pointer;color:var(--text-muted);transition:all .15s;border-left:3px solid transparent;font-size:13px;font-weight:400}
.nav-item:hover{color:var(--text);background:var(--surface-hover)}
.nav-item.active{color:var(--accent);background:var(--accent-muted);border-left-color:var(--accent);font-weight:500}
.nav-item svg{width:16px;height:16px;flex-shrink:0;opacity:.6}
.nav-item.active svg{opacity:1}
.main{margin-left:var(--sidebar-w);height:100vh;overflow-y:auto;padding:28px 32px 48px}
.main::-webkit-scrollbar{width:6px}
.main::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.page-header{margin-bottom:24px}
.page-header h2{font-size:22px;font-weight:700;letter-spacing:-.02em}
.page-header p{color:var(--text-muted);margin-top:4px;font-size:13px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}
.kpi-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;transition:all .2s;cursor:default;animation:fadeInUp .3s ease both}
.kpi-card:hover{transform:translateY(-2px);border-color:var(--border-light);box-shadow:0 8px 24px rgba(0,0,0,.3)}
.kpi-card:nth-child(1){animation-delay:.05s}
.kpi-card:nth-child(2){animation-delay:.1s}
.kpi-card:nth-child(3){animation-delay:.15s}
.kpi-card:nth-child(4){animation-delay:.2s}
.kpi-top{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.kpi-icon{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}
.kpi-icon.blue{background:var(--accent-muted);color:var(--accent)}
.kpi-icon.green{background:var(--success-muted);color:var(--success)}
.kpi-icon.purple{background:var(--purple-muted);color:var(--purple)}
.kpi-icon.yellow{background:var(--warning-muted);color:var(--warning)}
.kpi-icon.red{background:var(--danger-muted);color:var(--danger)}
.kpi-label{font-size:12px;color:var(--text-muted);font-weight:500;text-transform:uppercase;letter-spacing:.04em}
.kpi-value{font-family:'Sora',sans-serif;font-size:28px;font-weight:700;line-height:1.1}
.kpi-secondary{font-size:12px;color:var(--text-dim);margin-top:4px}
.grid-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:24px}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;animation:fadeInUp .3s ease both;transition:all .2s}
.card:hover{border-color:var(--border-light)}
.card-title{font-size:14px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.card-title svg{width:16px;height:16px;opacity:.6}
.info-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(30,58,95,.3)}
.info-row:last-child{border-bottom:none}
.info-label{font-size:12px;color:var(--text-muted);font-weight:500}
.info-value{font-size:13px;font-weight:500;text-align:right;max-width:60%;word-break:break-all}
.info-value.mono{font-family:'JetBrains Mono',monospace;font-size:12px}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:9999px;font-size:11px;font-weight:600;letter-spacing:.02em}
.badge-accent{background:var(--accent-muted);color:var(--accent)}
.badge-success{background:var(--success-muted);color:var(--success)}
.badge-warning{background:var(--warning-muted);color:var(--warning)}
.badge-danger{background:var(--danger-muted);color:var(--danger)}
.badge-purple{background:var(--purple-muted);color:var(--purple)}
.badge-muted{background:var(--surface-muted);color:var(--text-dim)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot-green{background:var(--success)}
.dot-gray{background:var(--text-dim)}
.dot-yellow{background:var(--warning)}
.dot-red{background:var(--danger)}
.dot-pulse{animation:pulse 2s infinite}
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:10px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--text-dim);border-bottom:1px solid var(--border);background:var(--surface-muted)}
td{padding:10px 12px;font-size:13px;border-bottom:1px solid rgba(30,58,95,.2)}
tr:hover td{background:var(--surface-hover)}
.provider-grid,.tool-grid,.memory-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.item-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;transition:all .2s;animation:fadeInUp .3s ease both}
.item-card:hover{transform:translateY(-1px);border-color:var(--border-light);box-shadow:0 4px 16px rgba(0,0,0,.2)}
.item-card.active-item{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.item-card .item-name{font-size:14px;font-weight:600;margin-bottom:4px}
.item-card .item-hint{font-size:12px;color:var(--text-muted);margin-bottom:8px;line-height:1.4}
.item-card .item-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.item-card .item-env{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-dim);margin-top:6px}
.section-title{font-size:16px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.section-title svg{width:20px;height:20px;flex-shrink:0}
.category-group{margin-bottom:24px}
.category-group h3{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.empty-state{text-align:center;padding:40px 20px;color:var(--text-dim)}
.empty-state svg{display:block;width:48px;height:48px;max-width:48px;max-height:48px;opacity:.3;margin:0 auto 12px}
.empty-state p{font-size:14px}
.code-block{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:16px;overflow-x:auto;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.6;white-space:pre-wrap;word-break:break-all;max-height:600px;overflow-y:auto}
.code-block::-webkit-scrollbar{width:4px;height:4px}
.code-block::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.json-key{color:var(--accent)}
.json-string{color:var(--success)}
.json-number{color:var(--warning)}
.json-bool{color:var(--purple)}
.json-null{color:var(--danger)}
.btn{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:6px;font-size:12px;font-weight:600;border:1px solid transparent;cursor:pointer;transition:all .15s;font-family:'IBM Plex Sans',sans-serif}
.btn-accent{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn-accent:hover{background:var(--accent-hover)}
.btn-danger{background:var(--danger-muted);color:var(--danger);border-color:var(--danger)}
.btn-danger:hover{background:var(--danger);color:#fff}
.btn-outline{background:transparent;color:var(--text-muted);border-color:var(--border)}
.btn-outline:hover{border-color:var(--border-light);color:var(--text)}
.gauge-container{display:flex;justify-content:center;padding:20px}
.gauge-container svg text{font-family:'Sora',sans-serif}
.neuro-bar{margin-bottom:14px}
.neuro-bar-header{display:flex;justify-content:space-between;margin-bottom:4px}
.neuro-bar-label{font-size:12px;font-weight:500}
.neuro-bar-value{font-size:12px;font-family:'Sora',sans-serif;font-weight:600}
.neuro-bar-track{height:8px;background:var(--surface-muted);border-radius:4px;overflow:hidden}
.neuro-bar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.approval-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:10px;animation:fadeInUp .3s ease both}
.approval-card .approval-actions{display:flex;gap:8px;margin-top:10px}
.event-item{padding:10px 0;border-bottom:1px solid rgba(30,58,95,.2);font-size:13px;display:flex;gap:10px;animation:fadeInUp .2s ease both}
.event-item .event-time{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-dim);white-space:nowrap;min-width:70px}
.event-item .event-body{flex:1}
.sse-status{display:flex;align-items:center;gap:6px;font-size:12px;margin-bottom:14px}
.security-levels{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.level-card{background:var(--surface);border:2px solid var(--border);border-radius:10px;padding:16px;text-align:center;cursor:default;transition:all .2s}
.level-card.active-level{border-color:var(--accent);box-shadow:0 0 12px rgba(59,130,246,.2)}
.level-card h4{font-size:14px;margin-bottom:4px}
.level-card p{font-size:11px;color:var(--text-muted)}
.hamburger{display:none;position:fixed;top:16px;left:16px;z-index:200;width:36px;height:36px;background:var(--surface);border:1px solid var(--border);border-radius:8px;cursor:pointer;align-items:center;justify-content:center}
.hamburger svg{width:20px;height:20px;color:var(--text)}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:90}
@media(max-width:768px){
.sidebar{transform:translateX(-100%)}
.sidebar.open{transform:translateX(0)}
.overlay.open{display:block}
.hamburger{display:flex}
.main{margin-left:0;padding:20px 16px 48px;padding-top:60px}
.kpi-grid{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.grid-3{grid-template-columns:1fr}
.grid-2{grid-template-columns:1fr}
}
.loading-shimmer{height:200px;border-radius:10px}
.tool-badge-list{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.tool-badge{font-family:'JetBrains Mono',monospace;font-size:10px;padding:2px 6px;border-radius:4px;background:var(--surface-muted);color:var(--text-muted)}
</style>
</head>
<body>
<div class="hamburger" id="hamburger"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg></div>
<div class="overlay" id="overlay"></div>
<aside class="sidebar" id="sidebar">
<div class="sidebar-header">
<svg width="32" height="32" viewBox="0 0 32 32" fill="none">
<rect width="32" height="32" rx="8" fill="#3b82f6"/>
<path d="M8 16c0-4.4 3.6-8 8-8s8 3.6 8 8" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/>
<path d="M12 16c0-2.2 1.8-4 4-4s4 1.8 4 4" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
<circle cx="16" cy="16" r="2" fill="#fff"/>
<path d="M16 18v6" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
<path d="M13 22h6" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
</svg>
<div>
<h1>ZeroClaw</h1>
<span>Mission Control</span>
</div>
</div>
<nav class="nav-scroll" id="nav"></nav>
</aside>
<main class="main" id="main"></main>
<script>
(function(){
var NAV = [
{id: 'overview', label: 'Overview', icon: 'grid', group: 'System'},
{id: 'providers', label: 'Providers', icon: 'cpu', group: 'Services'},
{id: 'channels', label: 'Channels', icon: 'radio', group: 'Services'},
{id: 'tunnels', label: 'Tunnels', icon: 'globe', group: 'Services'},
{id: 'memory', label: 'Memory', icon: 'database', group: 'Infrastructure'},
{id: 'tools', label: 'Tools', icon: 'wrench', group: 'Infrastructure'},
{id: 'observers', label: 'Observers', icon: 'eye', group: 'Infrastructure'},
{id: 'runtimes', label: 'Runtimes', icon: 'play', group: 'Infrastructure'},
{id: 'peripherals', label: 'Peripherals', icon: 'usb', group: 'Infrastructure'},
{id: 'bots', label: 'Bots', icon: 'bot', group: 'Control'},
{id: 'commands', label: 'Commands', icon: 'terminal', group: 'Control'},
{id: 'approvals', label: 'Approvals', icon: 'check', group: 'Control'},
{id: 'audit', label: 'Audit', icon: 'scroll', group: 'Control'},
{id: 'events', label: 'Events', icon: 'zap', group: 'Control'},
{id: 'consciousness', label: 'Consciousness', icon: 'brain', group: 'Intelligence'},
{id: 'security', label: 'Security', icon: 'shield', group: 'Settings'},
{id: 'config', label: 'Config', icon: 'settings', group: 'Settings'}
];

var IC = {
grid:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
cpu:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
radio:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 010 8.49M7.76 16.24a6 6 0 010-8.49M19.07 4.93a10 10 0 010 14.14M4.93 19.07a10 10 0 010-14.14"/></svg>',
globe:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10"/></svg>',
database:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
wrench:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>',
eye:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
play:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
usb:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22V8M5 12V8h14v4"/><circle cx="12" cy="5" r="3"/><rect x="7" y="15" width="4" height="4" rx="1"/><rect x="13" y="15" width="4" height="4" rx="1"/></svg>',
bot:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><circle cx="8" cy="16" r="1"/><circle cx="16" cy="16" r="1"/></svg>',
terminal:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
scroll:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 21h12a2 2 0 002-2v-2H10v2a2 2 0 11-4 0V5a2 2 0 00-2-2H2v2a2 2 0 002 2h4v14z"/></svg>',
zap:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
brain:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z"/><path d="M9 21h6M10 17v4M14 17v4"/></svg>',
shield:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
settings:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>'
};

var S = {currentPage:'overview',cache:{},eventSource:null,refreshTimer:null};

function esc(s){
if(s===null||s===undefined)return'';
return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

function sc(el,h){
var n=typeof el==='string'?document.getElementById(el):el;
if(n)n.innerHTML=h;
}

function fj(url){
return fetch(url).then(function(r){if(!r.ok)return null;return r.json();}).catch(function(){return null;});
}

function toggleSidebar(){
document.getElementById('sidebar').classList.toggle('open');
document.getElementById('overlay').classList.toggle('open');
}

document.getElementById('hamburger').addEventListener('click',toggleSidebar);
document.getElementById('overlay').addEventListener('click',toggleSidebar);

function buildNav(){
var g={};
NAV.forEach(function(s){if(!g[s.group])g[s.group]=[];g[s.group].push(s);});
var h='';
Object.keys(g).forEach(function(k){
h+='<div class="nav-group" data-group="'+esc(k)+'">';
h+='<div class="nav-group-label">'+esc(k)+'<span class="arrow">&#9660;</span></div>';
h+='<div class="nav-items">';
g[k].forEach(function(s){
h+='<div class="nav-item'+(s.id===S.currentPage?' active':'')+'" data-page="'+s.id+'">';
h+=(IC[s.icon]||'')+'<span>'+esc(s.label)+'</span></div>';
});
h+='</div></div>';
});
sc('nav',h);
document.querySelectorAll('.nav-group-label').forEach(function(el){
el.addEventListener('click',function(){this.parentElement.classList.toggle('collapsed');});
});
document.querySelectorAll('.nav-item').forEach(function(el){
el.addEventListener('click',function(){navigate(this.getAttribute('data-page'));});
});
}

function navigate(p){
if(S.refreshTimer){clearInterval(S.refreshTimer);S.refreshTimer=null;}
if(S.eventSource){S.eventSource.close();S.eventSource=null;}
S.currentPage=p;
window.location.hash=p;
buildNav();
document.getElementById('sidebar').classList.remove('open');
document.getElementById('overlay').classList.remove('open');
renderPage(p);
}

function renderPage(p){
var m=document.getElementById('main');
m.scrollTop=0;
var R={overview:rOverview,providers:rProviders,channels:rChannels,tools:rTools,memory:rMemory,observers:rObservers,runtimes:rRuntimes,tunnels:rTunnels,security:rSecurity,config:rConfig,bots:rBots,commands:rCommands,approvals:rApprovals,audit:rAudit,events:rEvents,consciousness:rConsciousness,peripherals:rPeripherals};
if(R[p])R[p]();
else sc(m,'<div class="empty-state"><p>Page not found</p></div>');
}

function catBadge(c){
var m={frontier:'badge-purple',aggregator:'badge-accent',local:'badge-success',inference:'badge-warning',search:'badge-accent',china:'badge-danger'};
return '<span class="badge '+(m[c]||'badge-muted')+'">'+esc(c)+'</span>';
}

function secBadge(l){
if(!l)return'<span class="badge badge-muted">Unknown</span>';
var v=String(l).toLowerCase();
if(v==='autonomous')return'<span class="badge badge-danger">'+esc(l)+'</span>';
if(v==='supervised')return'<span class="badge badge-warning">'+esc(l)+'</span>';
if(v==='locked')return'<span class="badge badge-success">'+esc(l)+'</span>';
return'<span class="badge badge-accent">'+esc(l)+'</span>';
}

function boolB(v){
if(v===true)return'<span class="badge badge-success">Yes</span>';
if(v===false)return'<span class="badge badge-muted">No</span>';
return'<span class="badge badge-muted">N/A</span>';
}

function fmtJSON(o,d){
if(o===null||o===undefined)return'<span class="json-null">null</span>';
d=d||0;
var p='  '.repeat(d),p1='  '.repeat(d+1);
if(typeof o==='string')return'<span class="json-string">&quot;'+esc(o)+'&quot;</span>';
if(typeof o==='number')return'<span class="json-number">'+o+'</span>';
if(typeof o==='boolean')return'<span class="json-bool">'+o+'</span>';
if(Array.isArray(o)){
if(!o.length)return'[]';
var items=o.map(function(v){return p1+fmtJSON(v,d+1);});
return'[\n'+items.join(',\n')+'\n'+p+']';
}
if(typeof o==='object'){
var k=Object.keys(o);
if(!k.length)return'{}';
var e=k.map(function(key){return p1+'<span class="json-key">&quot;'+esc(key)+'&quot;</span>: '+fmtJSON(o[key],d+1);});
return'{\n'+e.join(',\n')+'\n'+p+'}';
}
return esc(String(o));
}

function rOverview(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Overview</h2><p>System status and key metrics</p></div><div id="ov-kpi" class="kpi-grid"><div class="kpi-card shimmer loading-shimmer"></div><div class="kpi-card shimmer loading-shimmer"></div><div class="kpi-card shimmer loading-shimmer"></div><div class="kpi-card shimmer loading-shimmer"></div></div><div id="ov-info" class="grid-3"></div><div id="ov-approvals"></div><div id="ov-activity"></div>');

Promise.all([fj('/api/status'),fj('/api/system'),fj('/api/control/approvals'),fj('/api/control/audit')]).then(function(r){
var st=r[0],sy=r[1],ap=r[2],au=r[3];
if(ap&&!Array.isArray(ap)&&ap.approvals)ap=ap.approvals;
if(au&&!Array.isArray(au)&&au.entries)au=au.entries;
var cc=st?st.channels_count:0;
var cn=st&&st.channels?st.channels.join(', '):'';
var pv=st?(esc(st.provider)+' / '+esc(st.model)):'N/A';
var tc=st?st.tools_count:0;
var tt=sy&&sy.tools?(sy.tools.total||sy.tools.items.length):tc;
var sl=st&&st.security?st.security.autonomy_level:'Unknown';

sc('ov-kpi',
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon blue">&#9670;</div><div class="kpi-label">Active Channels</div></div><div class="kpi-value">'+cc+'</div><div class="kpi-secondary">'+esc(cn)+'</div></div>'+
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon purple">&#9729;</div><div class="kpi-label">Provider</div></div><div class="kpi-value" style="font-size:16px">'+pv+'</div><div class="kpi-secondary">Active model</div></div>'+
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon green">&#9881;</div><div class="kpi-label">Tools Enabled</div></div><div class="kpi-value">'+tc+'</div><div class="kpi-secondary">of '+tt+' total</div></div>'+
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon yellow">&#9888;</div><div class="kpi-label">Security Level</div></div><div class="kpi-value" style="font-size:18px">'+secBadge(sl)+'</div><div class="kpi-secondary">Autonomy mode</div></div>'
);

var sh='<div class="card fade-in"><div class="card-title">'+IC.settings+' System Status</div>';
if(st){
sh+='<div class="info-row"><span class="info-label">Gateway</span><span class="info-value mono">'+esc(st.gateway?st.gateway.host+':'+st.gateway.port:'N/A')+'</span></div>';
sh+='<div class="info-row"><span class="info-label">Memory Backend</span><span class="info-value">'+esc(st.memory_backend)+'</span></div>';
sh+='<div class="info-row"><span class="info-label">Runtime</span><span class="info-value">'+esc(sy&&sy.runtimes&&sy.runtimes.items.length?sy.runtimes.items[0].name:'default')+'</span></div>';
sh+='<div class="info-row"><span class="info-label">Observers</span><span class="info-value">'+(sy&&sy.observers?sy.observers.items.length:0)+'</span></div>';
sh+='<div class="info-row"><span class="info-label">Temperature</span><span class="info-value">'+st.temperature+'</span></div>';
sh+='<div class="info-row"><span class="info-label">Identity Format</span><span class="info-value">'+esc(st.identity?st.identity.format:'N/A')+'</span></div>';
}
sh+='</div>';

var seh='<div class="card fade-in"><div class="card-title">'+IC.shield+' Security</div>';
if(st&&st.security){
seh+='<div class="info-row"><span class="info-label">Autonomy Level</span><span class="info-value">'+secBadge(st.security.autonomy_level)+'</span></div>';
seh+='<div class="info-row"><span class="info-label">Sandbox</span><span class="info-value">'+boolB(st.security.sandbox_enabled)+'</span></div>';
var ws=sy&&sy.security?sy.security.workspace_only:null;
seh+='<div class="info-row"><span class="info-label">Workspace Only</span><span class="info-value">'+boolB(ws)+'</span></div>';
seh+='<div class="info-row"><span class="info-label">Pairing Required</span><span class="info-value">'+boolB(st.gateway?st.gateway.require_pairing:null)+'</span></div>';
var aa=sy&&sy.security?sy.security.auto_approve:[];
if(aa&&aa.length>0){
seh+='<div class="info-row"><span class="info-label">Auto-approved</span><span class="info-value"><div class="tool-badge-list">';
aa.forEach(function(t){seh+='<span class="tool-badge">'+esc(t)+'</span>';});
seh+='</div></span></div>';
}else{
seh+='<div class="info-row"><span class="info-label">Auto-approved</span><span class="info-value badge badge-muted">None</span></div>';
}
}
seh+='</div>';

var gh='<div class="card fade-in"><div class="card-title">'+IC.globe+' Gateway Health</div>';
if(st&&st.gateway){
gh+='<div class="info-row"><span class="info-label">Host</span><span class="info-value mono">'+esc(st.gateway.host)+'</span></div>';
gh+='<div class="info-row"><span class="info-label">Port</span><span class="info-value mono">'+st.gateway.port+'</span></div>';
gh+='<div class="info-row"><span class="info-label">Pairing Required</span><span class="info-value">'+boolB(st.gateway.require_pairing)+'</span></div>';
gh+='<div class="info-row"><span class="info-label">Agents</span><span class="info-value">'+esc(st.agents_count)+'</span></div>';
gh+='<div class="info-row"><span class="info-label">Uptime</span><span class="info-value"><span class="dot dot-green dot-pulse"></span> Online</span></div>';
}
gh+='</div>';

sc('ov-info',sh+seh+gh);

var ah='<div class="section-title" style="margin-top:8px">'+IC.check+' Pending Approvals</div>';
if(ap&&Array.isArray(ap)&&ap.length>0){
ap.forEach(function(a){
ah+='<div class="approval-card"><div style="font-weight:500">'+esc(a.description||a.action||a.id||'Approval request')+'</div>';
if(a.tool)ah+='<div style="font-size:12px;color:var(--text-muted);margin-top:4px">Tool: <span class="mono">'+esc(a.tool)+'</span></div>';
if(a.timestamp)ah+='<div style="font-size:11px;color:var(--text-dim);margin-top:2px">'+esc(a.timestamp)+'</div>';
ah+='<div class="approval-actions"><button class="btn btn-accent" data-approve="'+esc(a.id||'')+'">Approve</button><button class="btn btn-danger" data-deny="'+esc(a.id||'')+'">Deny</button></div></div>';
});
}else{
ah+='<div class="empty-state"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg><p>No pending approvals</p></div>';
}
sc('ov-approvals',ah);
bindApprovalButtons('ov-approvals');

var ach='<div class="section-title">'+IC.scroll+' Recent Activity</div>';
if(au&&Array.isArray(au)&&au.length>0){
var items=au.slice(-10).reverse();
items.forEach(function(e){
ach+='<div class="event-item"><span class="event-time">'+esc(e.timestamp||e.time||'')+'</span><span class="event-body">'+esc(e.action||e.message||e.event||JSON.stringify(e))+'</span></div>';
});
}else{
ach+='<div class="empty-state"><p>No recent activity</p></div>';
}
sc('ov-activity',ach);
});

S.refreshTimer=setInterval(function(){if(S.currentPage==='overview')rOverview();},30000);
}

function bindApprovalButtons(containerId){
var c=document.getElementById(containerId);
if(!c)return;
c.querySelectorAll('[data-approve]').forEach(function(b){
b.addEventListener('click',function(){approveAction(this.getAttribute('data-approve'));});
});
c.querySelectorAll('[data-deny]').forEach(function(b){
b.addEventListener('click',function(){denyAction(this.getAttribute('data-deny'));});
});
}

function approveAction(id){
fetch('/api/control/approvals/'+encodeURIComponent(id)+'/approve',{method:'POST'}).then(function(){
if(S.currentPage==='approvals')rApprovals();
else if(S.currentPage==='overview')rOverview();
}).catch(function(){});
}

function denyAction(id){
fetch('/api/control/approvals/'+encodeURIComponent(id)+'/deny',{method:'POST'}).then(function(){
if(S.currentPage==='approvals')rApprovals();
else if(S.currentPage==='overview')rOverview();
}).catch(function(){});
}

function rProviders(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Providers</h2><p>AI model providers and inference backends</p></div><div id="prov-grid" class="provider-grid"><div class="item-card shimmer loading-shimmer"></div></div>');
Promise.all([fj('/api/system'),fj('/api/status')]).then(function(r){
var sy=r[0],st=r[1];
if(!sy||!sy.providers||!sy.providers.items){sc('prov-grid','<div class="empty-state"><p>Unable to load providers</p></div>');return;}
var ap=st?st.provider:'';
var h='';
sy.providers.items.forEach(function(p,i){
var ia=p.name===ap;
h+='<div class="item-card'+(ia?' active-item':'')+'" style="animation-delay:'+(i*0.03)+'s">';
h+='<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">'+catBadge(p.category||'');
if(ia)h+='<span class="badge badge-success">Active</span>';
else h+=(p.enabled?'<span class="dot dot-green"></span>':'<span class="dot dot-gray"></span>');
h+='</div>';
h+='<div class="item-name">'+esc(p.label||p.name)+'</div>';
h+='<div class="item-hint">'+esc(p.hint||'')+'</div>';
if(p.env_var)h+='<div class="item-env">'+esc(p.env_var)+'</div>';
h+='</div>';
});
sc('prov-grid',h);
});
}

function rChannels(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Channels</h2><p>Communication interfaces</p></div><div class="card" id="ch-table"><div class="shimmer loading-shimmer"></div></div>');
Promise.all([fj('/api/channels'),fj('/api/status')]).then(function(r){
var raw=r[0],st=r[1];
var ch=Array.isArray(raw)?raw:(raw&&Array.isArray(raw.channels)?raw.channels:null);
if(!ch){sc('ch-table','<div class="empty-state"><p>Unable to load channels</p></div>');return;}
var ac=st&&st.channels?st.channels:[];
var h='<div class="card-title">'+IC.radio+' '+ac.length+' active of '+ch.length+' total</div>';
h+='<table><thead><tr><th>Status</th><th>Name</th><th>Category</th><th>Required Keys</th><th>Optional Keys</th><th>Hint</th></tr></thead><tbody>';
ch.forEach(function(c){
var ia=ac.indexOf(c.name)>=0;
h+='<tr><td><span class="dot '+(ia?'dot-green dot-pulse':'dot-gray')+'"></span></td>';
h+='<td style="font-weight:500">'+esc(c.label||c.name)+'</td>';
h+='<td>'+catBadge(c.category||'')+'</td>';
h+='<td class="mono" style="font-size:11px">'+esc((c.required_keys||[]).join(', ')||'None')+'</td>';
h+='<td class="mono" style="font-size:11px">'+esc((c.optional_keys||[]).join(', ')||'None')+'</td>';
h+='<td style="color:var(--text-muted);font-size:12px">'+esc(c.hint||'')+'</td></tr>';
});
h+='</tbody></table>';
sc('ch-table',h);
});
}

function rTools(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Tools</h2><p>Available capabilities</p></div><div id="tool-content"><div class="shimmer loading-shimmer"></div></div>');
Promise.all([fj('/api/system'),fj('/api/status')]).then(function(r){
var sy=r[0],st=r[1];
if(!sy||!sy.tools||!sy.tools.items){sc('tool-content','<div class="empty-state"><p>Unable to load tools</p></div>');return;}
var en=st&&st.tools_enabled?st.tools_enabled:[];
var g={};
sy.tools.items.forEach(function(t){var c=t.category||'uncategorized';if(!g[c])g[c]=[];g[c].push(t);});
var h='';
Object.keys(g).sort().forEach(function(c){
h+='<div class="category-group"><h3>'+esc(c)+' ('+g[c].length+')</h3><div class="tool-grid">';
g[c].forEach(function(t){
var ie=en.indexOf(t.name)>=0;
h+='<div class="item-card'+(ie?' active-item':'')+'">';
h+='<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">';
h+='<span class="item-name mono">'+esc(t.name)+'</span>';
if(ie)h+='<span class="badge badge-success">Enabled</span>';
h+='</div><div class="item-hint">'+esc(t.hint||'')+'</div></div>';
});
h+='</div></div>';
});
sc('tool-content',h);
});
}

function rItemsPage(title,sub,key,activeField,icon){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>'+esc(title)+'</h2><p>'+esc(sub)+'</p></div><div id="items-grid" class="memory-grid"><div class="item-card shimmer loading-shimmer"></div></div>');
Promise.all([fj('/api/system'),fj('/api/status')]).then(function(r){
var sy=r[0],st=r[1];
if(!sy||!sy[key]||!sy[key].items){sc('items-grid','<div class="empty-state"><p>No '+esc(key)+' data available</p></div>');return;}
var av=activeField&&st?st[activeField]:null;
var h='';
sy[key].items.forEach(function(item,i){
var ia=av?(item.name===av):item.enabled;
h+='<div class="item-card'+(ia?' active-item':'')+'" style="animation-delay:'+(i*0.05)+'s">';
h+='<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">';
h+='<span class="item-name">'+esc(item.label||item.name)+'</span>';
if(ia)h+='<span class="badge badge-success">Active</span>';
else h+=(item.enabled?'<span class="dot dot-green"></span>':'<span class="dot dot-gray"></span>');
h+='</div><div class="item-hint">'+esc(item.hint||'')+'</div>';
if(item.env_var)h+='<div class="item-env">'+esc(item.env_var)+'</div>';
h+='</div>';
});
sc('items-grid',h);
});
}

function rMemory(){rItemsPage('Memory','Storage backends','memory','memory_backend','database');}
function rObservers(){rItemsPage('Observers','System monitoring hooks','observers',null,'eye');}
function rRuntimes(){rItemsPage('Runtimes','Execution environments','runtimes',null,'play');}
function rTunnels(){rItemsPage('Tunnels','Network tunneling services','tunnels',null,'globe');}

function rSecurity(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Security</h2><p>Access control and autonomy settings</p></div><div id="sec-content"><div class="shimmer loading-shimmer"></div></div>');
Promise.all([fj('/api/system'),fj('/api/status')]).then(function(r){
var sy=r[0],st=r[1];
var cl=st&&st.security?st.security.autonomy_level:'';
var lv=sy&&sy.security&&sy.security.levels?sy.security.levels:[];
var aa=sy&&sy.security?(sy.security.auto_approve||[]):[];
var h='<div class="section-title">'+IC.shield+' Autonomy Levels</div><div class="security-levels">';
lv.forEach(function(l){
var nm=typeof l==='string'?l:(l.name||l.label||'');
var lb=typeof l==='string'?l:(l.label||l.name||'');
var ht=typeof l==='object'&&l.hint?l.hint:'';
var ia=(typeof l==='object'&&l.active===true)||(nm.toLowerCase()===cl.toLowerCase());
h+='<div class="level-card'+(ia?' active-level':'')+'"><h4>'+esc(lb)+'</h4>';
if(ht)h+='<p style="font-size:12px;color:var(--text-muted);margin-top:4px">'+esc(ht)+'</p>';
if(ia)h+='<p style="color:var(--accent);font-weight:600;margin-top:4px">Current level</p>';
h+='</div>';
});
h+='</div>';
h+='<div class="card" style="margin-bottom:16px"><div class="card-title">'+IC.shield+' Security Flags</div>';
h+='<div class="info-row"><span class="info-label">Sandbox Enabled</span><span class="info-value">'+boolB(st&&st.security?st.security.sandbox_enabled:null)+'</span></div>';
var ws=sy&&sy.security?sy.security.workspace_only:null;
h+='<div class="info-row"><span class="info-label">Workspace Only</span><span class="info-value">'+boolB(ws)+'</span></div>';
h+='<div class="info-row"><span class="info-label">Gateway Pairing</span><span class="info-value">'+boolB(st&&st.gateway?st.gateway.require_pairing:null)+'</span></div>';
h+='</div>';
h+='<div class="card"><div class="card-title">Auto-approved Tools</div>';
if(aa.length>0){
h+='<div class="tool-badge-list">';
aa.forEach(function(t){h+='<span class="tool-badge">'+esc(t)+'</span>';});
h+='</div>';
}else{
h+='<div style="color:var(--text-dim);font-size:13px">No tools auto-approved</div>';
}
h+='</div>';
sc('sec-content',h);
});
}

function rConfig(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Config</h2><p>Current runtime configuration</p></div><div id="cfg-content"><div class="shimmer loading-shimmer"></div></div>');
fj('/api/status').then(function(d){
if(!d){sc('cfg-content','<div class="empty-state"><p>Unable to load configuration</p></div>');return;}
sc('cfg-content','<div class="code-block">'+fmtJSON(d)+'</div>');
});
}

function rControlTable(title,sub,endpoint){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>'+esc(title)+'</h2><p>'+esc(sub)+'</p></div><div class="card" id="ctrl-table"><div class="shimmer loading-shimmer"></div></div>');
fj(endpoint).then(function(d){
if(d&&!Array.isArray(d)&&typeof d==='object'){var k=Object.keys(d).filter(function(x){return Array.isArray(d[x]);});if(k.length)d=d[k[0]];}
if(!d||!Array.isArray(d)||!d.length){sc('ctrl-table','<div class="empty-state"><p>No '+esc(title.toLowerCase())+' found</p></div>');return;}
var cols=Object.keys(d[0]);
var h='<table><thead><tr>';
cols.forEach(function(c){h+='<th>'+esc(c)+'</th>';});
h+='</tr></thead><tbody>';
d.forEach(function(row){
h+='<tr>';
cols.forEach(function(c){
var v=row[c];
if(typeof v==='object'&&v!==null)v=JSON.stringify(v);
h+='<td>'+esc(v)+'</td>';
});
h+='</tr>';
});
h+='</tbody></table>';
sc('ctrl-table',h);
});
}

function rBots(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Bots &amp; Connections</h2><p>All active connections, channels, and registered agents</p></div><div id="bots-kpi" class="kpi-grid"><div class="kpi-card shimmer loading-shimmer"></div><div class="kpi-card shimmer loading-shimmer"></div><div class="kpi-card shimmer loading-shimmer"></div></div><div id="bots-channels" class="card"><div class="shimmer loading-shimmer"></div></div><div id="bots-agents" class="card" style="margin-top:16px"><div class="shimmer loading-shimmer"></div></div>');
Promise.all([fj('/api/status'),fj('/api/channels'),fj('/api/control/bots')]).then(function(r){
var st=r[0],raw=r[1],bd=r[2];
var ch=raw&&Array.isArray(raw.channels)?raw.channels:(Array.isArray(raw)?raw:[]);
var ac=st&&st.channels?st.channels:[];
var bots=bd&&Array.isArray(bd.bots)?bd.bots:(Array.isArray(bd)?bd:[]);
var ag=st?st.agents_count:0;
sc('bots-kpi',
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon green">&#9889;</div><div class="kpi-label">Active Channels</div></div><div class="kpi-value">'+ac.length+'</div><div class="kpi-secondary">of '+ch.length+' available</div></div>'+
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon purple">&#9670;</div><div class="kpi-label">Registered Bots</div></div><div class="kpi-value">'+bots.length+'</div><div class="kpi-secondary">Control plane agents</div></div>'+
'<div class="kpi-card fade-in"><div class="kpi-top"><div class="kpi-icon blue">&#9729;</div><div class="kpi-label">Named Agents</div></div><div class="kpi-value">'+ag+'</div><div class="kpi-secondary">'+(st&&st.agents?st.agents.join(', '):'None')+'</div></div>'
);
var h='<div class="card-title">'+IC.radio+' Channel Connections</div>';
if(ch.length){
h+='<table><thead><tr><th>Status</th><th>Channel</th><th>Category</th><th>Required Keys</th><th>Setup Hint</th></tr></thead><tbody>';
ch.forEach(function(c){
var ia=ac.indexOf(c.name)>=0;
h+='<tr><td><span class="dot '+(ia?'dot-green dot-pulse':'dot-gray')+'"></span> '+(ia?'<span class="badge badge-success" style="margin-left:4px">Connected</span>':'<span class="badge badge-muted" style="margin-left:4px">Inactive</span>')+'</td>';
h+='<td style="font-weight:500">'+esc(c.label||c.name)+'</td>';
h+='<td>'+catBadge(c.category||'')+'</td>';
h+='<td class="mono" style="font-size:11px">'+esc((c.required_keys||[]).join(', ')||'None')+'</td>';
h+='<td style="color:var(--text-muted);font-size:12px">'+esc(c.hint||'')+'</td></tr>';
});
h+='</tbody></table>';
}else{h+='<div class="empty-state"><p>No channels configured</p></div>';}
sc('bots-channels',h);
var bh='<div class="card-title">'+IC.bot+' Registered Bots</div>';
if(bots.length){
bh+='<table><thead><tr><th>Status</th><th>Name</th><th>ID</th><th>Gateway URL</th><th>Provider</th><th>Channels</th><th>Memory</th><th>Workspace</th><th>Uptime</th><th>Last Heartbeat</th></tr></thead><tbody>';
bots.forEach(function(row){
var isOnline=row.status==='online';
var dot=isOnline?'dot-green dot-pulse':'dot-red';
var badge=isOnline?'<span class="badge badge-success">Online</span>':'<span class="badge badge-muted">Offline</span>';
var url='http://'+esc(row.host||'127.0.0.1')+':'+esc(''+row.port);
var uptimeMin=Math.floor((row.uptime_secs||0)/60);
bh+='<tr>';
bh+='<td><span class="dot '+dot+'"></span> '+badge+'</td>';
bh+='<td style="font-weight:500">'+esc(row.name||row.id)+'</td>';
bh+='<td class="mono" style="font-size:11px">'+esc(row.id)+'</td>';
bh+='<td><a href="'+url+'" target="_blank" style="color:var(--accent)">'+esc(url)+'</a></td>';
bh+='<td>'+esc(row.provider||'-')+'</td>';
bh+='<td>'+esc(row.channels||'[]')+'</td>';
bh+='<td>'+esc(row.memory_backend||'-')+'</td>';
bh+='<td class="mono" style="font-size:11px">'+esc(row.workspace_dir||'-')+'</td>';
bh+='<td>'+uptimeMin+'m</td>';
bh+='<td style="font-size:11px">'+esc(row.last_heartbeat||'-')+'</td>';
bh+='</tr>';
});
bh+='</tbody></table>';
}else{bh+='<div class="empty-state">'+IC.bot+'<p>No bots registered yet. Connect a channel to see bots here.</p></div>';}
sc('bots-agents',bh);
});
}
function rCommands(){rControlTable('Commands','Available control commands','/api/control/commands');}

function rApprovals(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Approvals</h2><p>Pending approval requests</p></div><div id="appr-content"><div class="shimmer loading-shimmer"></div></div>');
fj('/api/control/approvals').then(function(d){
if(d&&!Array.isArray(d)&&d.approvals)d=d.approvals;
if(!d||!Array.isArray(d)||!d.length){
sc('appr-content','<div class="empty-state"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg><p>No pending approvals</p></div>');
return;
}
var h='';
d.forEach(function(a,i){
h+='<div class="approval-card fade-in" style="animation-delay:'+(i*0.05)+'s">';
h+='<div style="display:flex;justify-content:space-between;align-items:flex-start">';
h+='<div><div style="font-weight:600;margin-bottom:4px">'+esc(a.description||a.action||a.id||'Request #'+(i+1))+'</div>';
if(a.tool)h+='<div style="font-size:12px;color:var(--text-muted)">Tool: <span class="mono">'+esc(a.tool)+'</span></div>';
if(a.agent)h+='<div style="font-size:12px;color:var(--text-muted)">Agent: '+esc(a.agent)+'</div>';
if(a.timestamp)h+='<div style="font-size:11px;color:var(--text-dim);margin-top:2px">'+esc(a.timestamp)+'</div>';
h+='</div>';
h+='<div class="approval-actions"><button class="btn btn-accent" data-approve="'+esc(a.id||'')+'">Approve</button><button class="btn btn-danger" data-deny="'+esc(a.id||'')+'">Deny</button></div>';
h+='</div></div>';
});
sc('appr-content',h);
bindApprovalButtons('appr-content');
});
}

function rAudit(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Audit</h2><p>System event log</p></div><div class="card" id="audit-content"><div class="shimmer loading-shimmer"></div></div>');
fj('/api/control/audit').then(function(d){
if(d&&!Array.isArray(d)&&d.entries)d=d.entries;
if(!d||!Array.isArray(d)||!d.length){sc('audit-content','<div class="empty-state"><p>No audit entries</p></div>');return;}
var h='';
d.slice().reverse().forEach(function(e,i){
h+='<div class="event-item fade-in" style="animation-delay:'+(i*0.02)+'s">';
h+='<span class="event-time">'+esc(e.timestamp||e.time||'')+'</span>';
h+='<span class="event-body">';
if(e.level){
var lc=(e.level||'').toLowerCase();
var bc=lc==='error'?'badge-danger':lc==='warn'?'badge-warning':'badge-muted';
h+='<span class="badge '+bc+'" style="margin-right:6px">'+esc(e.level)+'</span>';
}
h+=esc(e.action||e.message||e.event||JSON.stringify(e));
if(e.agent)h+=' <span style="color:var(--text-dim)">by '+esc(e.agent)+'</span>';
h+='</span></div>';
});
sc('audit-content',h);
});
}

function rEvents(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Events</h2><p>Live event stream</p></div><div id="sse-status" class="sse-status"><span class="dot dot-yellow dot-pulse"></span> Connecting...</div><div class="card" id="event-feed" style="max-height:600px;overflow-y:auto"></div>');
sc('event-feed','');
if(S.eventSource){S.eventSource.close();}
try{
S.eventSource=new EventSource('/api/control/events/stream');
S.eventSource.onopen=function(){
sc('sse-status','<span class="dot dot-green dot-pulse"></span> Connected');
};
S.eventSource.onmessage=function(e){
var feed=document.getElementById('event-feed');
if(!feed)return;
var now=new Date().toLocaleTimeString();
var div=document.createElement('div');
div.className='event-item fade-in';
var timeSpan=document.createElement('span');
timeSpan.className='event-time';
timeSpan.textContent=now;
var bodySpan=document.createElement('span');
bodySpan.className='event-body';
bodySpan.textContent=e.data;
div.appendChild(timeSpan);
div.appendChild(bodySpan);
feed.insertBefore(div,feed.firstChild);
if(feed.children.length>200)feed.removeChild(feed.lastChild);
};
S.eventSource.onerror=function(){
sc('sse-status','<span class="dot dot-red"></span> Disconnected — retrying...');
};
}catch(err){
sc('sse-status','<span class="dot dot-red"></span> SSE not available');
}
}

function rConsciousness(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Consciousness</h2><p>Real-time neural correlates and phenomenal state</p></div>'+
'<div class="sse-status" id="con-ws-status"><span class="dot dot-gray dot-pulse"></span> Connecting...</div>'+
'<div class="kpi-grid" id="con-kpis">'+
'<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon blue">'+IC.brain+'</div><div><div class="kpi-label">Coherence</div></div></div><div class="kpi-value" id="con-coh">--</div><div class="kpi-secondary" id="con-coh-bar"><div class="neuro-bar-track" style="margin-top:6px"><div class="neuro-bar-fill" id="con-coh-fill" style="width:0%;background:var(--accent);transition:width .4s"></div></div></div></div>'+
'<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon green">'+IC.zap+'</div><div><div class="kpi-label">Tick</div></div></div><div class="kpi-value mono" id="con-tick">--</div></div>'+
'<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon purple">'+IC.check+'</div><div><div class="kpi-label">Proposals</div></div></div><div class="kpi-value" id="con-props">--</div><div class="kpi-secondary" id="con-props-detail"></div></div>'+
'<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon yellow">'+IC.settings+'</div><div><div class="kpi-label">Debate Rounds</div></div></div><div class="kpi-value" id="con-debate">--</div></div>'+
'</div>'+
'<div class="grid-2">'+
'<div class="card" id="con-neuro-card"><div class="card-title">'+IC.brain+' Neuromodulators</div><div id="con-neuro"></div></div>'+
'<div class="card" id="con-phenom-card"><div class="card-title">'+IC.eye+' Phenomenal State</div>'+
'<div style="display:flex;gap:16px;flex-wrap:wrap" id="con-phenom"></div></div>'+
'</div>'+
'<div class="card" style="margin-top:16px" id="con-ncn-card"><div class="card-title">'+IC.settings+' NCN Control Signals</div><div id="con-ncn"></div></div>'
);
var ws,reconDelay=1000;
function mkBar(label,pct,color){
var d=document.createElement('div');d.className='neuro-bar';
var hd=document.createElement('div');hd.className='neuro-bar-header';
var lbl=document.createElement('span');lbl.className='neuro-bar-label';lbl.textContent=label;
var val=document.createElement('span');val.className='neuro-bar-value';val.style.color=color;val.textContent=pct+'%';
hd.appendChild(lbl);hd.appendChild(val);
var trk=document.createElement('div');trk.className='neuro-bar-track';
var fl=document.createElement('div');fl.className='neuro-bar-fill';fl.style.width=pct+'%';fl.style.background=color;fl.style.transition='width .4s';
trk.appendChild(fl);d.appendChild(hd);d.appendChild(trk);return d;
}
function drawNeuro(nd){
var el=document.getElementById('con-neuro');if(!el)return;
el.textContent='';
var nm=[{k:'dopamine',l:'Dopamine',c:'var(--accent)'},{k:'serotonin',l:'Serotonin',c:'var(--success)'},{k:'norepinephrine',l:'Norepinephrine',c:'var(--warning)'},{k:'cortisol',l:'Cortisol',c:'var(--danger)'}];
nm.forEach(function(n){var v=nd&&typeof nd[n.k]==='number'?nd[n.k]:0;var p=v>1?v:Math.round(v*100);
el.appendChild(mkBar(n.l,p,n.c));});
}
function drawPhenom(ph){
var el=document.getElementById('con-phenom');if(!el)return;
el.textContent='';
var keys=[{k:'attention',c:'var(--accent)'},{k:'arousal',c:'var(--warning)'},{k:'valence',c:'var(--success)'}];
keys.forEach(function(item){var v=ph&&typeof ph[item.k]==='number'?ph[item.k]:0;var p=Math.round(v*100);
var wrap=document.createElement('div');wrap.style.cssText='flex:1;min-width:80px;text-align:center';
var num=document.createElement('div');num.style.cssText='font-size:28px;font-weight:700;color:'+item.c+';font-family:Sora,sans-serif';num.textContent=p+'%';
var lbl=document.createElement('div');lbl.style.cssText='font-size:11px;color:var(--text-muted);margin-top:2px;text-transform:uppercase;letter-spacing:.04em';lbl.textContent=item.k;
var trk=document.createElement('div');trk.className='neuro-bar-track';trk.style.marginTop='6px';
var fl=document.createElement('div');fl.className='neuro-bar-fill';fl.style.width=p+'%';fl.style.background=item.c;fl.style.transition='width .4s';
trk.appendChild(fl);wrap.appendChild(num);wrap.appendChild(lbl);wrap.appendChild(trk);el.appendChild(wrap);});
}
function drawNcn(ncn){
var el=document.getElementById('con-ncn');if(!el)return;
el.textContent='';
var keys=[{k:'precision',l:'Precision',c:'var(--accent)'},{k:'gain',l:'Gain',c:'var(--purple)'},{k:'ffn_gate',l:'FFN Gate',c:'var(--warning)'}];
keys.forEach(function(n){var v=ncn&&typeof ncn[n.k]==='number'?ncn[n.k]:0;var p=Math.round(v*100);
el.appendChild(mkBar(n.l,p,n.c));});
}
function updateUI(d){
var ce=document.getElementById('con-coh');if(ce)ce.textContent=typeof d.coherence==='number'?(d.coherence*100).toFixed(1)+'%':'--';
var cf=document.getElementById('con-coh-fill');if(cf)cf.style.width=(typeof d.coherence==='number'?Math.round(d.coherence*100):0)+'%';
var te=document.getElementById('con-tick');if(te)te.textContent=d.tick_count!==undefined?d.tick_count:'--';
var pe=document.getElementById('con-props');if(pe)pe.textContent=d.last_tick_approved!==undefined?d.last_tick_approved+'/'+d.last_tick_proposals:'--';
var pd=document.getElementById('con-props-detail');if(pd&&d.last_tick_vetoed!==undefined)pd.textContent=d.last_tick_vetoed+' vetoed';
var de=document.getElementById('con-debate');if(de&&d.debate_rounds_used!==undefined)de.textContent=d.debate_rounds_used;
drawNeuro(d.modulators);
drawPhenom(d.phenomenal);
drawNcn(d.ncn_signals);
}
function connectWS(){
var proto=location.protocol==='https:'?'wss:':'ws:';
ws=new WebSocket(proto+'//'+location.host+'/api/consciousness/stream');
ws.onopen=function(){
var s=document.getElementById('con-ws-status');
if(s){s.textContent='';var dot=document.createElement('span');dot.className='dot dot-green dot-pulse';s.appendChild(dot);s.appendChild(document.createTextNode(' Connected (live)'));}
reconDelay=1000;
};
ws.onmessage=function(e){try{var d=JSON.parse(e.data);updateUI(d);}catch(x){}};
ws.onclose=function(){
var s=document.getElementById('con-ws-status');
if(s){s.textContent='';var dot=document.createElement('span');dot.className='dot dot-yellow dot-pulse';s.appendChild(dot);s.appendChild(document.createTextNode(' Reconnecting...'));}
setTimeout(connectWS,reconDelay);
reconDelay=Math.min(reconDelay*2,30000);
};
ws.onerror=function(){ws.close();};
}
connectWS();
drawNeuro(null);drawPhenom(null);drawNcn(null);
}

function rPeripherals(){
var m=document.getElementById('main');
sc(m,'<div class="page-header"><h2>Peripherals</h2><p>Connected hardware devices</p></div><div id="periph-content"><div class="shimmer loading-shimmer"></div></div>');
fj('/api/system').then(function(d){
if(!d||!d.peripherals||!Array.isArray(d.peripherals.items)||!d.peripherals.items.length){
sc('periph-content','<div class="empty-state"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22V8M5 12V8h14v4"/><circle cx="12" cy="5" r="3"/><rect x="7" y="15" width="4" height="4" rx="1"/><rect x="13" y="15" width="4" height="4" rx="1"/></svg><p>No peripherals connected</p></div>');
return;
}
var h='<div class="memory-grid">';
d.peripherals.items.forEach(function(p,i){
h+='<div class="item-card fade-in" style="animation-delay:'+(i*0.05)+'s">';
h+='<div class="item-name">'+esc(p.label||p.name)+'</div>';
h+='<div class="item-hint">'+esc(p.hint||p.type||'')+'</div>';
if(p.status)h+='<div style="margin-top:6px">'+boolB(p.status==='connected')+'</div>';
h+='</div>';
});
h+='</div>';
sc('periph-content',h);
});
}

window.navigate=navigate;

function init(){
var hash=window.location.hash.replace('#','');
var valid=NAV.map(function(s){return s.id;});
if(hash&&valid.indexOf(hash)>=0)S.currentPage=hash;
buildNav();
renderPage(S.currentPage);
window.addEventListener('hashchange',function(){
var h=window.location.hash.replace('#','');
if(h&&valid.indexOf(h)>=0&&h!==S.currentPage)navigate(h);
});
}

init();
})();
</script>
</body>
</html>"##;

const VALID_PROVIDERS: &[&str] = &[
    "openai",
    "anthropic",
    "google",
    "ollama",
    "groq",
    "mistral",
    "cohere",
    "deepseek",
    "xai",
    "openrouter",
    "fireworks",
    "together",
    "perplexity",
    "aws_bedrock",
    "azure",
    "cloudflare_ai",
    "cerebras",
    "sambanova",
    "hyperbolic",
    "lmstudio",
    "custom",
];

const VALID_MEMORY_BACKENDS: &[&str] = &["sqlite", "lucid", "postgres", "markdown", "none"];

const VALID_OBSERVER_BACKENDS: &[&str] = &["none", "log", "prometheus", "otel"];

const VALID_RUNTIME_KINDS: &[&str] = &["native", "docker", "wasm"];

const VALID_TUNNEL_PROVIDERS: &[&str] = &["none", "cloudflare", "tailscale", "ngrok", "custom"];

const SECRET_FIELDS: &[&str] = &[
    "bot_token",
    "app_token",
    "app_secret",
    "api_key",
    "access_token",
    "client_secret",
    "server_password",
    "nickserv_password",
    "sasl_password",
    "password",
    "verify_token",
    "encrypt_key",
    "verification_token",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_html_is_valid_document() {
        let html = DASHBOARD_HTML.trim();
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.ends_with("</html>"));
        assert!(DASHBOARD_HTML.contains("<head>"));
        assert!(DASHBOARD_HTML.contains("</head>"));
        assert!(DASHBOARD_HTML.contains("<body"));
        assert!(DASHBOARD_HTML.contains("</body>"));
    }

    #[test]
    fn dashboard_html_contains_zeroclaw_branding() {
        assert!(DASHBOARD_HTML.contains("ZeroClaw"));
        assert!(DASHBOARD_HTML.contains("Dashboard"));
    }

    #[test]
    fn dashboard_html_references_all_api_endpoints() {
        let expected_endpoints = [
            "/api/system",
            "/api/channels",
            "/api/status",
            "/api/control/",
            "/api/consciousness",
        ];
        for ep in &expected_endpoints {
            assert!(
                DASHBOARD_HTML.contains(ep),
                "Dashboard HTML missing endpoint reference: {ep}"
            );
        }
    }

    #[test]
    fn dashboard_html_has_all_nav_sections() {
        let nav_sections = [
            "overview",
            "providers",
            "channels",
            "tools",
            "memory",
            "observers",
            "runtimes",
            "security",
            "tunnels",
            "config",
            "bots",
            "commands",
            "approvals",
            "audit",
            "events",
            "consciousness",
            "peripherals",
        ];
        for section in &nav_sections {
            let nav_id = format!("id: '{section}'");
            assert!(
                DASHBOARD_HTML.contains(&nav_id),
                "Dashboard HTML missing nav section: {section}"
            );
        }
    }

    #[test]
    fn valid_providers_contains_major_providers() {
        let required = ["openai", "anthropic", "ollama", "openrouter", "deepseek"];
        for p in &required {
            assert!(
                VALID_PROVIDERS.contains(p),
                "Missing required provider: {p}"
            );
        }
        assert_eq!(VALID_PROVIDERS.len(), 21);
    }

    #[test]
    fn valid_memory_backends_are_complete() {
        assert!(VALID_MEMORY_BACKENDS.contains(&"sqlite"));
        assert!(VALID_MEMORY_BACKENDS.contains(&"none"));
        assert_eq!(VALID_MEMORY_BACKENDS.len(), 5);
    }

    #[test]
    fn valid_observer_backends_are_complete() {
        assert!(VALID_OBSERVER_BACKENDS.contains(&"none"));
        assert!(VALID_OBSERVER_BACKENDS.contains(&"prometheus"));
        assert_eq!(VALID_OBSERVER_BACKENDS.len(), 4);
    }

    #[test]
    fn valid_runtime_kinds_are_complete() {
        assert!(VALID_RUNTIME_KINDS.contains(&"native"));
        assert_eq!(VALID_RUNTIME_KINDS.len(), 3);
    }

    #[test]
    fn valid_tunnel_providers_are_complete() {
        assert!(VALID_TUNNEL_PROVIDERS.contains(&"none"));
        assert!(VALID_TUNNEL_PROVIDERS.contains(&"cloudflare"));
        assert_eq!(VALID_TUNNEL_PROVIDERS.len(), 5);
    }

    #[test]
    fn secret_fields_covers_sensitive_keys() {
        let critical = [
            "bot_token",
            "access_token",
            "api_key",
            "password",
            "client_secret",
        ];
        for field in &critical {
            assert!(
                SECRET_FIELDS.contains(field),
                "Missing secret field: {field}"
            );
        }
        assert!(SECRET_FIELDS.len() >= 10);
    }

    #[test]
    fn no_duplicate_entries_in_validation_lists() {
        fn has_duplicates<'a>(items: &'a [&'a str]) -> Option<&'a str> {
            let mut seen = std::collections::HashSet::new();
            items.iter().find(|&item| !seen.insert(item)).copied()
        }

        assert_eq!(
            has_duplicates(VALID_PROVIDERS),
            None,
            "Duplicate in VALID_PROVIDERS"
        );
        assert_eq!(
            has_duplicates(VALID_MEMORY_BACKENDS),
            None,
            "Duplicate in VALID_MEMORY_BACKENDS"
        );
        assert_eq!(
            has_duplicates(VALID_OBSERVER_BACKENDS),
            None,
            "Duplicate in VALID_OBSERVER_BACKENDS"
        );
        assert_eq!(
            has_duplicates(VALID_RUNTIME_KINDS),
            None,
            "Duplicate in VALID_RUNTIME_KINDS"
        );
        assert_eq!(
            has_duplicates(VALID_TUNNEL_PROVIDERS),
            None,
            "Duplicate in VALID_TUNNEL_PROVIDERS"
        );
        assert_eq!(
            has_duplicates(SECRET_FIELDS),
            None,
            "Duplicate in SECRET_FIELDS"
        );
    }

    #[test]
    fn dashboard_html_does_not_contain_inline_secrets() {
        let forbidden = ["sk-", "xoxb-", "ghp_", "AKIA", "password123"];
        for pattern in &forbidden {
            assert!(
                !DASHBOARD_HTML.contains(pattern),
                "Dashboard HTML contains potential secret pattern: {pattern}"
            );
        }
    }
}
