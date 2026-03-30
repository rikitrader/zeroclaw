use std::collections::HashMap;
use std::time::Duration;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct PhaseStats {
    pub count: usize,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileSummary {
    pub total_ticks: usize,
    pub avg_tick_ms: f64,
    pub p50_tick_ms: f64,
    pub p99_tick_ms: f64,
    pub phase_breakdown: HashMap<String, PhaseStats>,
    pub agent_breakdown: HashMap<String, PhaseStats>,
    pub memory_trend: Vec<usize>,
}

pub struct ConsciousnessProfiler {
    tick_durations: Vec<Duration>,
    phase_durations: HashMap<String, Vec<Duration>>,
    agent_durations: HashMap<String, Vec<Duration>>,
    memory_snapshots: Vec<usize>,
}

impl ConsciousnessProfiler {
    pub fn new() -> Self {
        Self {
            tick_durations: Vec::new(),
            phase_durations: HashMap::new(),
            agent_durations: HashMap::new(),
            memory_snapshots: Vec::new(),
        }
    }

    pub fn record_tick(&mut self, duration: Duration) {
        self.tick_durations.push(duration);
    }

    pub fn record_phase(&mut self, phase: &str, duration: Duration) {
        self.phase_durations
            .entry(phase.to_string())
            .or_default()
            .push(duration);
    }

    pub fn record_agent(&mut self, agent: &str, duration: Duration) {
        self.agent_durations
            .entry(agent.to_string())
            .or_default()
            .push(duration);
    }

    pub fn record_memory_snapshot(&mut self, bytes: usize) {
        if self.memory_snapshots.len() > 1000 {
            self.memory_snapshots.drain(..500);
        }
        self.memory_snapshots.push(bytes);
    }

    pub fn summary(&self) -> ProfileSummary {
        ProfileSummary {
            total_ticks: self.tick_durations.len(),
            avg_tick_ms: avg_duration_ms(&self.tick_durations),
            p50_tick_ms: percentile_ms(&self.tick_durations, 50),
            p99_tick_ms: percentile_ms(&self.tick_durations, 99),
            phase_breakdown: self
                .phase_durations
                .iter()
                .map(|(k, v)| (k.clone(), compute_stats(v)))
                .collect(),
            agent_breakdown: self
                .agent_durations
                .iter()
                .map(|(k, v)| (k.clone(), compute_stats(v)))
                .collect(),
            memory_trend: self.memory_snapshots.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.tick_durations.clear();
        self.phase_durations.clear();
        self.agent_durations.clear();
        self.memory_snapshots.clear();
    }
}

impl Default for ConsciousnessProfiler {
    fn default() -> Self {
        Self::new()
    }
}

fn avg_duration_ms(durations: &[Duration]) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    let total: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    total / durations.len() as f64
}

fn percentile_ms(durations: &[Duration], pct: usize) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = (pct as f64 / 100.0 * (sorted.len() - 1) as f64)
        .round()
        .max(0.0) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_stats(durations: &[Duration]) -> PhaseStats {
    if durations.is_empty() {
        return PhaseStats {
            count: 0,
            avg_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p50_ms: 0.0,
            p99_ms: 0.0,
        };
    }
    let ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let min = ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max = ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    PhaseStats {
        count: durations.len(),
        avg_ms: avg_duration_ms(durations),
        min_ms: min,
        max_ms: max,
        p50_ms: percentile_ms(durations, 50),
        p99_ms: percentile_ms(durations, 99),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_records_and_summarizes() {
        let mut profiler = ConsciousnessProfiler::new();
        for i in 1..=10 {
            profiler.record_tick(Duration::from_millis(i));
            profiler.record_phase("perceive", Duration::from_micros(i * 100));
            profiler.record_phase("deliberate", Duration::from_micros(i * 200));
        }
        profiler.record_agent("chairman", Duration::from_micros(500));

        let summary = profiler.summary();
        assert_eq!(summary.total_ticks, 10);
        assert!(summary.avg_tick_ms > 0.0);
        assert!(summary.p50_tick_ms > 0.0);
        assert!(summary.p99_tick_ms >= summary.p50_tick_ms);
        assert!(summary.phase_breakdown.contains_key("perceive"));
        assert!(summary.phase_breakdown.contains_key("deliberate"));
        assert!(summary.agent_breakdown.contains_key("chairman"));
    }

    #[test]
    fn profiler_reset_clears_data() {
        let mut profiler = ConsciousnessProfiler::new();
        profiler.record_tick(Duration::from_millis(5));
        profiler.reset();
        let summary = profiler.summary();
        assert_eq!(summary.total_ticks, 0);
    }

    #[test]
    fn empty_profiler_summary() {
        let profiler = ConsciousnessProfiler::new();
        let summary = profiler.summary();
        assert_eq!(summary.total_ticks, 0);
        assert_eq!(summary.avg_tick_ms, 0.0);
    }
}
