use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle};

use crate::agents::process::agent_process;
use crate::config::provider::ProviderVariant;
use crate::constant::CORE_MD;
use crate::dependencies::VizierDependencies;
use crate::embedding::VizierEmbedder;
use crate::indexer::VizierIndexer;
use crate::indexer::sqlite::SqliteIndexer;
use crate::schema::{
    AgentCommand, AgentCommandResult, AgentConfig, AgentHealthState, AgentHealthStatus, AgentId,
    AgentSummary, ProviderEntryConfig, RevisionOrigin, RevisionTrigger,
};
use crate::storage::memory::MemoryStorage;
use crate::storage::agent::AgentStorage;
use crate::storage::provider::ProviderStorage;
use crate::storage::user::UserStorage;
use crate::utils::agent_workspace;

pub mod agent;
pub mod background;
pub mod hook;
pub mod mcp;
pub mod memory_ops;
pub mod process;
pub mod shell;
pub mod skill;
pub mod tools;

struct AgentProcess {
    handle: JoinHandle<()>,
    shutdown: watch::Sender<bool>,
    memory_op_handle: Option<JoinHandle<()>>,
}

struct StartingAgent {
    generation: u64,
    abort: AbortHandle,
    result: oneshot::Receiver<anyhow::Result<AgentProcess>>,
    response: Option<PendingStartupResponse>,
}

struct PendingStartupResponse {
    sender: oneshot::Sender<AgentCommandResult>,
    summary: AgentSummary,
    failure_prefix: &'static str,
}

enum ManagedAgent {
    Starting(StartingAgent),
    Running(AgentProcess),
    Failed,
}

struct AbortOnDrop(Option<JoinHandle<()>>);

impl AbortOnDrop {
    fn new(handle: JoinHandle<()>) -> Self {
        Self(Some(handle))
    }

    fn take(&mut self) -> JoinHandle<()> {
        self.0.take().expect("task handle already taken")
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
}

struct StartupCompletion {
    agent_id: AgentId,
    generation: u64,
}

pub struct VizierAgents {
    deps: VizierDependencies,
    processes: HashMap<AgentId, ManagedAgent>,
    startup_tx: mpsc::UnboundedSender<StartupCompletion>,
    startup_rx: mpsc::UnboundedReceiver<StartupCompletion>,
    next_generation: u64,
}

impl VizierAgents {
    pub async fn new(deps: VizierDependencies) -> Result<Self> {
        let (startup_tx, startup_rx) = mpsc::unbounded_channel();
        let mut agents = Self {
            deps,
            processes: HashMap::new(),
            startup_tx,
            startup_rx,
            next_generation: 0,
        };
        let stored_agents = agents.deps.storage.list_agents().await?;

        for (agent_id, config) in stored_agents {
            tracing::info!("starting agent: {}", agent_id);
            agents.start_agent(&agent_id, &config, None);
        }

        Ok(agents)
    }

    fn start_agent(
        &mut self,
        agent_id: &str,
        config: &AgentConfig,
        response: Option<PendingStartupResponse>,
    ) {
        self.next_generation += 1;
        let generation = self.next_generation;
        let agent_id_owned = agent_id.to_string();
        let deps = self.deps.clone();
        let config = config.clone();
        let startup =
            tokio::spawn(async move { Self::spawn_agent(&deps, &agent_id_owned, &config).await });
        let abort = startup.abort_handle();
        let (result_tx, result_rx) = oneshot::channel();
        let completion_tx = self.startup_tx.clone();
        let completion_agent_id = agent_id.to_string();

        tokio::spawn(async move {
            let result = match startup.await {
                Ok(result) => result,
                Err(error) => Err(anyhow::anyhow!("agent startup task failed: {error}")),
            };
            if let Err(Ok(process)) = result_tx.send(result) {
                Self::shutdown_process(process).await;
                return;
            }
            let _ = completion_tx.send(StartupCompletion {
                agent_id: completion_agent_id,
                generation,
            });
        });

        self.processes.insert(
            agent_id.to_string(),
            ManagedAgent::Starting(StartingAgent {
                generation,
                abort,
                result: result_rx,
                response,
            }),
        );
    }

    async fn finish_startup(&mut self, completion: StartupCompletion) {
        let Some(state) = self.processes.remove(&completion.agent_id) else {
            return;
        };
        let ManagedAgent::Starting(mut starting) = state else {
            self.processes.insert(completion.agent_id, state);
            return;
        };
        if starting.generation != completion.generation {
            self.processes
                .insert(completion.agent_id, ManagedAgent::Starting(starting));
            return;
        }

        let result = starting.result.try_recv().unwrap_or_else(|error| {
            Err(anyhow::anyhow!("agent startup result unavailable: {error}"))
        });
        match result {
            Ok(process) => {
                self.processes
                    .insert(completion.agent_id, ManagedAgent::Running(process));
                if let Some(response) = starting.response {
                    let _ = response
                        .sender
                        .send(AgentCommandResult::Ok(response.summary));
                }
            }
            Err(error) => {
                tracing::error!("failed to start agent '{}': {error}", completion.agent_id);
                self.processes
                    .insert(completion.agent_id, ManagedAgent::Failed);
                if let Some(response) = starting.response {
                    let _ = response.sender.send(AgentCommandResult::Error(format!(
                        "{}: {}",
                        response.failure_prefix, error
                    )));
                }
            }
        }
    }

    async fn stop_agent(&mut self, agent_id: &str) {
        match self.processes.remove(agent_id) {
            Some(ManagedAgent::Starting(mut starting)) => {
                starting.abort.abort();
                if let Ok(Ok(process)) = starting.result.await {
                    Self::shutdown_process(process).await;
                }
                if let Some(response) = starting.response.take() {
                    let _ = response.sender.send(AgentCommandResult::Error(
                        "agent startup was superseded by a newer lifecycle command".into(),
                    ));
                }
            }
            Some(ManagedAgent::Running(process)) => Self::shutdown_process(process).await,
            Some(ManagedAgent::Failed) | None => {}
        }

        self.deps
            .transport
            .unregister_agent(&agent_id.to_string())
            .await;
        self.deps
            .transport
            .unregister_memory_op(&agent_id.to_string())
            .await;
    }

    async fn shutdown_process(mut process: AgentProcess) {
        const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

        let _ = process.shutdown.send(true);
        if tokio::time::timeout(SHUTDOWN_TIMEOUT, &mut process.handle)
            .await
            .is_err()
        {
            process.handle.abort();
            let _ = process.handle.await;
        }
        if let Some(memory_op_handle) = process.memory_op_handle {
            memory_op_handle.abort();
            let _ = memory_op_handle.await;
        }
    }

    async fn build_indexer(
        deps: &VizierDependencies,
        config: &AgentConfig,
    ) -> Result<Option<VizierIndexer>> {
        let (emb_settings, _idx_cfg) = match (&config.embedding, &config.indexer) {
            (Some(e), Some(i)) => (e, i),
            _ => return Ok(None),
        };

        let conn = deps.sqlite_conn.clone();

        let embedder = Arc::new(
            VizierEmbedder::from_agent_settings(
                emb_settings,
                &deps.storage,
                &deps.config.workspace,
            )
            .await?,
        );
        let sqlite_idx = SqliteIndexer::new(conn, embedder).await?;
        Ok(Some(VizierIndexer::build(sqlite_idx)))
    }

    async fn spawn_agent(
        deps: &VizierDependencies,
        agent_id: &str,
        config: &AgentConfig,
    ) -> Result<AgentProcess> {
        if config.provider == ProviderVariant::ollama {
            if let Ok(Some(provider)) = deps.storage.get_provider(&ProviderVariant::ollama).await {
                if let ProviderEntryConfig::Ollama { base_url } = &provider.config {
                    crate::utils::ollama::ollama_pull_model(base_url, &config.model).await?;
                }
            }
        }

        let indexer = Self::build_indexer(deps, config).await?;
        let mut memory_op_handle = if let Some(idx) = indexer.clone() {
            let rx = deps
                .transport
                .register_memory_op(agent_id.to_string())
                .await;
            let storage = (*deps.storage).clone();
            let agent_id_owned = agent_id.to_string();
            let settings = memory_ops::MemoryOpSettings {
                chunking: config.chunking.clone(),
                per_document: config.auto_context.per_document,
            };
            Some(AbortOnDrop::new(tokio::spawn(async move {
                if let Err(e) =
                    memory_ops::handle_memory_ops(rx, idx, agent_id_owned, storage, settings).await
                {
                    tracing::error!("memory_ops handler exited with error: {}", e);
                }
            })))
        } else {
            None
        };

        // One-time passage conversion for this agent's existing corpus (FR-038).
        //
        // It lives here, and not in `dependencies.rs` where every other migration lives, because
        // **the indexer does not exist there**: it is built per agent from that agent's own
        // embedding config, which is why the two memory migrations in `dependencies.rs` pass
        // `NoopIndexer` and leave semantic search to catch up (research Decision 8). It must also
        // use *this* agent's indexer rather than any shared one, since two agents configured with
        // different embedding models write into one fixed-dimension vector table.
        //
        // Spawned detached rather than awaited, so the agent serves requests throughout (FR-040).
        // It holds neither the agent's request channel nor the sqlite write lock for the duration:
        // `reconcile_all_passages` takes the connection lock per statement and yields between
        // bundles.
        if let Some(idx) = indexer.clone() {
            let storage = deps.storage.clone();
            let agent_id_owned = agent_id.to_string();
            let limits = config.chunking.clone();
            tokio::spawn(async move {
                let started = std::time::Instant::now();
                match storage
                    .reconcile_agent_passages(&agent_id_owned, &limits, &idx)
                    .await
                {
                    // FR-042: counts processed, counts failed, and the reason for each failure —
                    // the reasons are logged per document by the reconcile itself. Silent when
                    // there was nothing to do, which is every start after the first.
                    Ok((0, 0)) => {}
                    Ok((converted, failed)) => tracing::info!(
                        agent_id = agent_id_owned,
                        converted,
                        failed,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "memory passage conversion finished"
                    ),
                    Err(e) => tracing::warn!(
                        agent_id = agent_id_owned,
                        "memory passage conversion failed; memories stay readable and are retried \
                         on the next start: {e}"
                    ),
                }
            });
        }

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let deps_clone = deps.clone();
        let agent_id_clone = agent_id.to_string();
        let config_clone = config.clone();
        let indexer_clone = indexer.clone();
        let (ready_tx, ready_rx) = oneshot::channel();

        let mut handle = AbortOnDrop::new(tokio::spawn(async move {
            if let Err(e) = agent_process(
                agent_id_clone.clone(),
                deps_clone,
                config_clone,
                indexer_clone,
                shutdown_rx,
                ready_tx,
            )
            .await
            {
                tracing::error!("agent '{}' exited with error: {}", agent_id_clone, e);
            }
        }));

        tokio::select! {
            result = ready_rx => {
                result.map_err(|_| anyhow::anyhow!("agent exited before becoming ready"))?;
            }
            _ = handle.0.as_mut().expect("agent task handle is present") => {
                anyhow::bail!("agent exited before becoming ready");
            }
        }

        Ok(AgentProcess {
            handle: handle.take(),
            shutdown: shutdown_tx,
            memory_op_handle: memory_op_handle.take().map(|mut handle| handle.take()),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                result = self.deps.transport.recv_agent_command() => {
                    match result? {
                        AgentCommand::Create {
                            agent_id,
                            config,
                            resp,
                        } => {
                            self.handle_create(&agent_id, config, resp).await;
                        }
                        AgentCommand::Update {
                            agent_id,
                            config,
                            resp,
                        } => {
                            self.handle_update(&agent_id, config, resp).await;
                        }
                        AgentCommand::Delete {
                            agent_id,
                            delete_workspace,
                            resp,
                        } => {
                            let result = self.handle_delete(&agent_id, delete_workspace).await;
                            let _ = resp.send(result);
                        }
                        AgentCommand::HealthCheck { resp } => {
                            let statuses = self
                                .processes
                                .iter()
                                .map(|(id, state)| {
                                    let (alive, status) = match state {
                                        ManagedAgent::Starting(_) => {
                                            (false, AgentHealthState::Starting)
                                        }
                                        ManagedAgent::Running(process)
                                            if !process.handle.is_finished() =>
                                        {
                                            (true, AgentHealthState::Running)
                                        }
                                        ManagedAgent::Running(_) | ManagedAgent::Failed => {
                                            (false, AgentHealthState::Failed)
                                        }
                                    };
                                    AgentHealthStatus {
                                        agent_id: id.clone(),
                                        alive,
                                        status,
                                    }
                                })
                                .collect();
                            let _ = resp.send(statuses);
                        }
                    }
                }
                Some(completion) = self.startup_rx.recv() => {
                    self.finish_startup(completion).await;
                }
            }
        }
    }

    async fn handle_create(
        &mut self,
        agent_id: &str,
        config: AgentConfig,
        response: oneshot::Sender<AgentCommandResult>,
    ) {
        if let Err(msg) = validate_config(&config) {
            let _ = response.send(AgentCommandResult::Error(msg));
            return;
        }

        if self.processes.contains_key(agent_id) {
            let _ = response.send(AgentCommandResult::Error(format!(
                "agent '{}' already exists",
                agent_id
            )));
            return;
        }

        if let Err(e) = self.deps.storage.create_agent(agent_id, &config).await {
            let _ = response.send(AgentCommandResult::Error(format!(
                "failed to persist agent: {}",
                e
            )));
            return;
        }

        if let Err(e) = self
            .deps
            .storage
            .set_agent_core(
                agent_id,
                CORE_MD,
                &RevisionOrigin::system(RevisionTrigger::Baseline),
            )
            .await
        {
            tracing::warn!(
                "failed to seed default CORE for agent '{}': {}",
                agent_id,
                e
            );
        }

        let summary = self.agent_summary(agent_id, &config).await;
        self.start_agent(
            agent_id,
            &config,
            Some(PendingStartupResponse {
                sender: response,
                summary,
                failure_prefix: "failed to start agent",
            }),
        );
    }

    async fn handle_update(
        &mut self,
        agent_id: &str,
        config: AgentConfig,
        response: oneshot::Sender<AgentCommandResult>,
    ) {
        if let Err(msg) = validate_config(&config) {
            let _ = response.send(AgentCommandResult::Error(msg));
            return;
        }

        if self
            .deps
            .storage
            .get_agent(agent_id)
            .await
            .ok()
            .flatten()
            .is_none()
        {
            let _ = response.send(AgentCommandResult::Error(format!(
                "agent '{}' not found",
                agent_id
            )));
            return;
        }

        if let Err(e) = self.deps.storage.update_agent(agent_id, &config).await {
            let _ = response.send(AgentCommandResult::Error(format!(
                "failed to persist agent: {}",
                e
            )));
            return;
        }

        self.stop_agent(agent_id).await;
        let summary = self.agent_summary(agent_id, &config).await;
        self.start_agent(
            agent_id,
            &config,
            Some(PendingStartupResponse {
                sender: response,
                summary,
                failure_prefix: "failed to restart agent",
            }),
        );
    }

    async fn agent_summary(&self, agent_id: &str, config: &AgentConfig) -> AgentSummary {
        let owner_username = if let Some(ref owner_id) = config.owner_id {
            self.deps
                .storage
                .get_user_by_id(owner_id)
                .await
                .ok()
                .flatten()
                .map(|user| user.username)
        } else {
            None
        };
        AgentSummary {
            agent_id: agent_id.to_string(),
            name: config.name.clone(),
            description: config.description.clone(),
            avatar_url: config.avatar_url.clone(),
            owner_username,
            owner_id: config.owner_id.clone(),
            shared_to: config.shared_to.clone(),
        }
    }

    async fn handle_delete(
        &mut self,
        agent_id: &str,
        delete_workspace: bool,
    ) -> AgentCommandResult {
        if self
            .deps
            .storage
            .get_agent(agent_id)
            .await
            .ok()
            .flatten()
            .is_none()
        {
            return AgentCommandResult::Error(format!("agent '{}' not found", agent_id));
        }

        if let Err(e) = self.deps.storage.delete_agent(agent_id).await {
            return AgentCommandResult::Error(format!("failed to delete agent: {}", e));
        }

        self.stop_agent(agent_id).await;

        if delete_workspace {
            let workspace = agent_workspace(&self.deps.config.workspace, agent_id);
            if workspace.exists() {
                if let Err(e) = std::fs::remove_dir_all(&workspace) {
                    tracing::warn!("failed to delete workspace for '{}': {}", agent_id, e);
                }
            }
        }

        AgentCommandResult::Ok(AgentSummary {
            agent_id: agent_id.to_string(),
            name: String::new(),
            description: None,
            avatar_url: None,
            owner_username: None,
            owner_id: None,
            shared_to: Vec::new(),
        })
    }
}

/// Invariants an agent config must satisfy before it is persisted.
fn validate_config(config: &AgentConfig) -> Result<(), String> {
    if config.tools.python.code_mode && !config.tools.python.enabled {
        return Err("tools.python.code_mode requires tools.python.enabled".into());
    }
    Ok(())
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use crate::{
        config::{VizierConfig, provider::ProviderVariant},
        dependencies::VizierDependencies,
        schema::AgentCommand,
        storage::provider::ProviderStorage,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn agent_config(provider: ProviderVariant, model: &str, name: &str) -> AgentConfig {
        AgentConfig {
            name: name.into(),
            owner_id: None,
            shared_to: vec![],
            system_prompt: None,
            description: None,
            provider,
            model: model.into(),
            thinking_depth: 5,
            checkpoint_threshold: 0.8,
            tools: Default::default(),
            silent_read_initiative_chance: 0.0,
            max_tokens: None,
            context_window: None,
            include_documents: None,
            prompt_timeout: Default::default(),
            documents: vec![],
            heartbeat_interval: Default::default(),
            core: None,
            dream_enabled: false,
            dream_schedule: None,
            dream_provider: None,
            dream_model: None,
            discord_token: None,
            telegram_token: None,
            avatar_url: None,
            embedding: None,
            indexer: None,
            chunking: Default::default(),
            auto_context: Default::default(),
        }
    }

    async fn send_agent_command(
        transport: &crate::transport::VizierTransport,
        command: impl FnOnce(oneshot::Sender<AgentCommandResult>) -> AgentCommand,
    ) -> AgentCommandResult {
        let (response_tx, response_rx) = oneshot::channel();
        transport
            .send_agent_command(command(response_tx))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), response_rx)
            .await
            .expect("manager did not answer agent command")
            .unwrap()
    }

    async fn health_statuses(
        transport: &crate::transport::VizierTransport,
    ) -> Vec<AgentHealthStatus> {
        let (response_tx, response_rx) = oneshot::channel();
        transport
            .send_agent_command(AgentCommand::HealthCheck { resp: response_tx })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), response_rx)
            .await
            .expect("health check was blocked by agent startup")
            .unwrap()
    }

    #[tokio::test]
    async fn slow_model_pull_does_not_block_commands_and_update_cancels_startup() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let (request_seen_tx, request_seen_rx) = oneshot::channel();
        let mock_ollama = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            loop {
                let count = socket.read(&mut buffer).await.unwrap();
                if count == 0 {
                    return;
                }
                request.extend_from_slice(&buffer[..count]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\r\n{\"status\":\"pulling\"}\n",
                )
                .await
                .unwrap();
            let _ = request_seen_tx.send(());
            tokio::time::sleep(Duration::from_secs(10)).await;
            let _ = socket.write_all(b"{\"status\":\"success\"}\n").await;
        });

        let directory = tempfile::tempdir().unwrap();
        let config = VizierConfig {
            workspace: directory.path().to_string_lossy().into_owned(),
            ..VizierConfig::default()
        };
        let deps = VizierDependencies::new(config).await.unwrap();
        deps.storage
            .upsert_provider(&crate::schema::ProviderEntry {
                variant: ProviderVariant::ollama,
                config: ProviderEntryConfig::Ollama {
                    base_url: base_url.clone(),
                },
            })
            .await
            .unwrap();
        let transport = deps.transport.clone();
        let mut manager = VizierAgents::new(deps).await.unwrap();
        let manager_task = tokio::spawn(async move { manager.run().await });

        let slow_config = agent_config(ProviderVariant::ollama, "mock:model", "slow");
        let (slow_response_tx, mut slow_response_rx) = oneshot::channel();
        transport
            .send_agent_command(AgentCommand::Create {
                agent_id: "slow".into(),
                config: slow_config,
                resp: slow_response_tx,
            })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), request_seen_rx)
            .await
            .expect("mock Ollama pull was not requested")
            .unwrap();

        let statuses = health_statuses(&transport).await;
        assert!(statuses.iter().any(|status| {
            status.agent_id == "slow"
                && status.status == AgentHealthState::Starting
                && !status.alive
        }));
        assert!(matches!(
            slow_response_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        let other_config = agent_config(ProviderVariant::dummyplug, "dummyplug", "other");
        let result = send_agent_command(&transport, |resp| AgentCommand::Create {
            agent_id: "other".into(),
            config: other_config,
            resp,
        })
        .await;
        assert!(matches!(result, AgentCommandResult::Ok(_)));

        let updated_config = agent_config(ProviderVariant::dummyplug, "dummyplug", "updated");
        let result = send_agent_command(&transport, |resp| AgentCommand::Update {
            agent_id: "slow".into(),
            config: updated_config,
            resp,
        })
        .await;
        assert!(matches!(result, AgentCommandResult::Ok(_)));
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), slow_response_rx)
                .await
                .unwrap()
                .unwrap(),
            AgentCommandResult::Error(_)
        ));

        let started = tokio::time::Instant::now();
        loop {
            let statuses = health_statuses(&transport).await;
            if statuses.iter().any(|status| {
                status.agent_id == "slow" && status.status == AgentHealthState::Running
            }) {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "updated agent did not start"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        for agent_id in ["slow", "other"] {
            let result = send_agent_command(&transport, |resp| AgentCommand::Delete {
                agent_id: agent_id.into(),
                delete_workspace: false,
                resp,
            })
            .await;
            assert!(matches!(result, AgentCommandResult::Ok(_)));
            assert!(!transport.is_agent_registered(&agent_id.to_string()).await);
        }

        manager_task.abort();
        let _ = manager_task.await;
        mock_ollama.abort();
    }
}
