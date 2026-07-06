pub mod docker;
pub mod native;
pub mod traits;
#[cfg(feature = "runtime-wasm")]
pub mod wasm;

pub use docker::DockerRuntime;
pub use native::NativeRuntime;
pub use traits::RuntimeAdapter;
#[cfg(feature = "runtime-wasm")]
pub use wasm::WasmRuntime;

use crate::config::RuntimeConfig;

/// Factory: create the right runtime from config
pub fn create_runtime(config: &RuntimeConfig) -> anyhow::Result<Box<dyn RuntimeAdapter>> {
    match config.kind {
        zeroclaw_config::schema::RuntimeKind::Native => Ok(Box::new(NativeRuntime::new())),
        zeroclaw_config::schema::RuntimeKind::Docker => {
            Ok(Box::new(DockerRuntime::new(config.docker.clone())))
        }
        #[cfg(feature = "runtime-wasm")]
        // V3 `RuntimeConfig` dropped the per-runtime `wasm` field; the WASM
        // runtime now starts from its own defaults.
        zeroclaw_config::schema::RuntimeKind::Wasm => Ok(Box::new(WasmRuntime::new(
            crate::config::WasmRuntimeConfig::default(),
        ))),
        #[cfg(not(feature = "runtime-wasm"))]
        zeroclaw_config::schema::RuntimeKind::Wasm => anyhow::bail!(
            "runtime.kind='wasm' requires the `runtime-wasm` feature. Rebuild with: cargo build --features runtime-wasm"
        ),
        zeroclaw_config::schema::RuntimeKind::Cloudflare => anyhow::bail!(
            "runtime.kind='cloudflare' is not implemented yet. Use runtime.kind='native' for now."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_native() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Native,
            ..RuntimeConfig::default()
        };
        let rt = create_runtime(&cfg).unwrap();
        assert_eq!(rt.name(), "native");
        assert!(rt.has_shell_access());
    }

    #[test]
    fn factory_docker() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Docker,
            ..RuntimeConfig::default()
        };
        let rt = create_runtime(&cfg).unwrap();
        assert_eq!(rt.name(), "docker");
        assert!(rt.has_shell_access());
    }

    #[test]
    fn factory_cloudflare_errors() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Cloudflare,
            ..RuntimeConfig::default()
        };
        match create_runtime(&cfg) {
            Err(err) => assert!(err.to_string().contains("not implemented")),
            Ok(_) => panic!("cloudflare runtime should error"),
        }
    }

    #[test]
    fn factory_created_runtime_is_correct_type() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Native,
            ..RuntimeConfig::default()
        };
        let rt = create_runtime(&cfg).unwrap();
        assert_eq!(rt.name(), "native");
        assert!(rt.has_shell_access());
        assert!(rt.has_filesystem_access());
        assert!(rt.supports_long_running());
    }

    #[test]
    fn factory_docker_created_from_config() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Docker,
            ..RuntimeConfig::default()
        };
        let rt = create_runtime(&cfg).unwrap();
        assert_eq!(rt.name(), "docker");
    }

    #[test]
    fn factory_adapters_impl_runtime_trait() {
        let native_cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Native,
            ..RuntimeConfig::default()
        };
        let docker_cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Docker,
            ..RuntimeConfig::default()
        };

        let native_rt = create_runtime(&native_cfg).unwrap();
        let docker_rt = create_runtime(&docker_cfg).unwrap();

        assert_eq!(native_rt.name(), "native");
        assert_eq!(docker_rt.name(), "docker");

        assert!(native_rt.has_shell_access());
        assert!(docker_rt.has_shell_access());
    }

    #[test]
    fn factory_returns_boxed_trait_object() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Native,
            ..RuntimeConfig::default()
        };
        let rt: Box<dyn crate::runtime::RuntimeAdapter> = create_runtime(&cfg).unwrap();
        assert_eq!(rt.name(), "native");
    }

    #[test]
    fn factory_cloudflare_message_clear() {
        let cfg = RuntimeConfig {
            kind: zeroclaw_config::schema::RuntimeKind::Cloudflare,
            ..RuntimeConfig::default()
        };
        match create_runtime(&cfg) {
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("cloudflare"));
                assert!(msg.contains("not implemented"));
            }
            Ok(_) => panic!("cloudflare should error"),
        }
    }
}
