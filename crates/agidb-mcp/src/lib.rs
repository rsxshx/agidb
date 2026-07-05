//! agidb MCP server — a stdio JSON-RPC server exposing the agidb engine
//! as MCP tools to Claude Desktop, Cursor, and other MCP-compatible
//! agents. Exposed tools:
//!
//! - `memory_observe` — record an observation
//! - `memory_recall` — tiered retrieval
//! - `memory_consolidate` — run one consolidation pass
//! - `memory_get_episode` — fetch one episode
//! - `memory_set_goal` — create a first-class goal
//! - `memory_active_goals` — list active goals
//! - `memory_assert_belief` — assert a revisable belief
//! - `memory_revise_belief` — revise a belief with episode evidence
//! - `memory_beliefs` — list beliefs
//! - `memory_unlearn` — cascading unlearn with audit + 30-day restore window
//! - `memory_what_did_i_learn` — introspect the learning log
//! - `memory_stats` — store-wide counts
//! - `memory_sense` — floor-1 sensory frame with surprise-gated promotion
//!
//! Phase 5 of the agidb v2 build. See
//! `docs/phases/phase-5-mcp-python.md`.
//!
//! Layout:
//! - [`protocol`] — JSON-RPC + MCP message types (parsing, errors).
//! - [`context`] — `AgidbContext`: Store + Extractor wrapper, the surface tools dispatch through.
//! - [`tools`] — tool registry + per-tool schema + handler.
//! - [`server`] — `McpServer`: pure `handle_request` + stdio driver.

pub mod context;
pub mod protocol;
pub mod server;
pub mod tools;

pub use crate::context::{AgidbContext, AgidbExtractor};
pub use crate::server::McpServer;
