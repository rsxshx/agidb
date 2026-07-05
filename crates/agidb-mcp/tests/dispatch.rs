//! End-to-end MCP server dispatch tests. Pure: drive `handle_request`
//! with JSON-RPC values, assert on the response. No stdio, no models.

use serde_json::{json, Value};

use agidb_mcp::protocol::{JsonRpcRequest, MCP_PROTOCOL_VERSION};
use agidb_mcp::{AgidbContext, McpServer};
use tempfile::TempDir;

fn fresh_server() -> (McpServer, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let ctx = AgidbContext::open_null(dir.path().to_str().unwrap()).expect("open");
    (McpServer::new(ctx), dir)
}

fn req(method: &str, params: Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: method.into(),
        params: Some(params),
    }
}

#[test]
fn initialize_returns_protocol_version_and_server_info() {
    let (server, _d) = fresh_server();
    let resp = server
        .handle_request(req("initialize", json!({})))
        .expect("response");
    let result = resp.result.expect("result");
    assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
    assert_eq!(result["serverInfo"]["name"], "agidb-mcp");
    assert!(result["capabilities"]["tools"].is_object());
}

#[test]
fn notifications_initialized_returns_none() {
    let (server, _d) = fresh_server();
    let mut r = req("notifications/initialized", json!({}));
    r.id = None;
    assert!(server.handle_request(r).is_none());
}

#[test]
fn tools_list_returns_the_four_registered_tools() {
    let (server, _d) = fresh_server();
    let resp = server
        .handle_request(req("tools/list", json!({})))
        .expect("response");
    let tools = resp.result.expect("result")["tools"]
        .as_array()
        .expect("tools array")
        .clone();
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("name"))
        .collect();
    assert!(names.contains(&"memory_observe"));
    assert!(names.contains(&"memory_recall"));
    assert!(names.contains(&"memory_consolidate"));
    assert!(names.contains(&"memory_get_episode"));
    // Each tool exposes a JSON-Schema input shape.
    for tool in &tools {
        assert_eq!(tool["inputSchema"]["type"], "object");
    }
}

#[test]
fn unknown_method_returns_method_not_found() {
    let (server, _d) = fresh_server();
    let resp = server
        .handle_request(req("nonsense/method", json!({})))
        .expect("response");
    let err = resp.error.expect("error");
    assert_eq!(err.code, -32600); // INVALID_REQUEST (we use that for unknown methods)
}

#[test]
fn observe_then_get_episode_roundtrips() {
    let (server, _d) = fresh_server();
    let observe = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_observe", "arguments": { "text": "Sarah recommended Bawri" } }),
        ))
        .expect("response");
    let observe_result = observe.result.expect("result");
    // The tool result wraps JSON-stringified payload in a content block.
    let content_text = observe_result["content"][0]["text"]
        .as_str()
        .expect("text content");
    let observe_payload: Value = serde_json::from_str(content_text).expect("inner json");
    let id = observe_payload["episode_id"].as_u64().expect("episode_id");
    assert!(id > 0);

    let fetched = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_get_episode", "arguments": { "id": id } }),
        ))
        .expect("response");
    let fetched_text = fetched.result.expect("result")["content"][0]["text"]
        .as_str()
        .expect("text content")
        .to_string();
    let fetched_payload: Value = serde_json::from_str(&fetched_text).expect("inner json");
    assert_eq!(fetched_payload["id"], json!(id));
    assert_eq!(fetched_payload["text"], "Sarah recommended Bawri");
}

#[test]
fn recall_after_observe_returns_a_match() {
    let (server, _d) = fresh_server();
    // Observe two things first so recall has something to find.
    for text in ["Sarah recommended Bawri", "Alice met Bob"] {
        server
            .handle_request(req(
                "tools/call",
                json!({ "name": "memory_observe", "arguments": { "text": text } }),
            ))
            .expect("observe");
    }
    let recall = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_recall", "arguments": { "cue": "Sarah", "k": 5 } }),
        ))
        .expect("response");
    let text = recall.result.expect("result")["content"][0]["text"]
        .as_str()
        .expect("text")
        .to_string();
    let payload: Value = serde_json::from_str(&text).expect("inner");
    // Per article VI, recall never returns empty under the default tier_floor.
    assert!(!payload["matches"]
        .as_array()
        .expect("matches array")
        .is_empty());
}

#[test]
fn invalid_tool_args_return_typed_error() {
    let (server, _d) = fresh_server();
    let resp = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_observe", "arguments": { "not_text": "oops" } }),
        ))
        .expect("response");
    let err = resp.error.expect("error");
    // memory_observe requires `text`; missing field is an InvalidParams.
    assert_eq!(err.code, -32602);
}

fn tool_payload(resp: serde_json::Value) -> Value {
    let text = resp["content"][0]["text"]
        .as_str()
        .expect("text content")
        .to_string();
    serde_json::from_str(&text).expect("inner json")
}

#[test]
fn goal_belief_unlearn_stats_tools_round_trip() {
    let (server, _d) = fresh_server();

    // set_goal → goal_id
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_set_goal", "arguments": { "description": "find a thai place for the team dinner" } }),
        ))
        .expect("set_goal");
    let v = tool_payload(r.result.expect("result"));
    let goal_id = v["goal_id"].as_u64().expect("goal_id");
    assert!(goal_id >= 1);

    // active_goals lists it
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_active_goals", "arguments": {} }),
        ))
        .expect("active_goals");
    let v = tool_payload(r.result.expect("result"));
    assert_eq!(v["goals"].as_array().unwrap().len(), 1);

    // assert_belief → belief_id
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_assert_belief", "arguments": { "claim": "Sarah likes thai food", "confidence": 0.8 } }),
        ))
        .expect("assert_belief");
    let v = tool_payload(r.result.expect("result"));
    let belief_id = v["belief_id"].as_u64().expect("belief_id");

    // observe an episode to use as revision evidence
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_observe", "arguments": { "text": "Sarah ordered pad thai again" } }),
        ))
        .expect("observe");
    let episode_id = tool_payload(r.result.expect("result"))["episode_id"]
        .as_u64()
        .unwrap();

    // revise_belief (supporting evidence) → confidence rises
    let r = server
        .handle_request(req(
            "tools/call",
            json!({
                "name": "memory_revise_belief",
                "arguments": {
                    "belief_id": belief_id,
                    "evidence_episode_id": episode_id,
                    "supports": true,
                    "reason": "she ordered it again"
                }
            }),
        ))
        .expect("revise_belief");
    let v = tool_payload(r.result.expect("result"));
    assert!(
        v["new_confidence"].as_f64().unwrap() > v["previous_confidence"].as_f64().unwrap(),
        "supporting evidence must raise confidence"
    );

    // beliefs list
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_beliefs", "arguments": {} }),
        ))
        .expect("beliefs");
    assert_eq!(
        tool_payload(r.result.expect("result"))["beliefs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    // stats
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_stats", "arguments": {} }),
        ))
        .expect("stats");
    assert_eq!(
        tool_payload(r.result.expect("result"))["episodes"]
            .as_u64()
            .unwrap(),
        1
    );

    // what_did_i_learn (default window) is non-empty
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_what_did_i_learn", "arguments": {} }),
        ))
        .expect("what_did_i_learn");
    assert!(!tool_payload(r.result.expect("result"))["events"]
        .as_array()
        .unwrap()
        .is_empty());

    // unlearn the episode
    let r = server
        .handle_request(req(
            "tools/call",
            json!({
                "name": "memory_unlearn",
                "arguments": {
                    "target_kind": "episode",
                    "target": episode_id.to_string(),
                    "reason": "test forget"
                }
            }),
        ))
        .expect("unlearn");
    let v = tool_payload(r.result.expect("result"));
    assert_eq!(v["episodes_removed"].as_u64().unwrap(), 1);
}

#[test]
fn sense_tool_promotes_novel_text() {
    let (server, _d) = fresh_server();
    let r = server
        .handle_request(req(
            "tools/call",
            json!({ "name": "memory_sense", "arguments": { "text": "novel sensory frame about the tokyo offsite" } }),
        ))
        .expect("sense");
    let v = tool_payload(r.result.expect("result"));
    assert!(v["surprise"].as_f64().unwrap() >= 0.4);
    assert!(v["promoted"].as_u64().is_some());
}
