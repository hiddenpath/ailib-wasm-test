//! # core_logic — Pure Rust protocol logic, testable on native target.
//!
//! 核心协议逻辑 — 纯 Rust 实现，不依赖 wasm-bindgen，可在原生目标上测试。
//!
//! Each function returns `Result<..., String>` so the wasm_bindgen wrappers
//! only need to convert `String` → `JsValue`.

use ai_lib_core::drivers::{OpenAiDriver, ProviderDriver};
use ai_lib_core::error_code::StandardErrorCode;
use ai_lib_core::protocol::v2::capabilities::Capability;
use ai_lib_core::types::message::Message;
use serde_json::Value;

/// Build a chat completion request body using ai-lib-core's OpenAiDriver.
pub fn build_chat_request(
    messages_json: &str,
    model: &str,
    temperature: f64,
    max_tokens: f64,
    stream: bool,
) -> Result<(String, bool), String> {
    let messages: Vec<Value> =
        serde_json::from_str(messages_json).map_err(|e| format!("Invalid messages JSON: {}", e))?;

    let ai_messages: Vec<Message> = messages
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
            match role {
                "system" => Message::system(content),
                "assistant" => Message::assistant(content),
                _ => Message::user(content),
            }
        })
        .collect();

    let driver = OpenAiDriver::new(
        "wasm-browser",
        vec![Capability::Text, Capability::Streaming],
    );
    let request = driver
        .build_request(
            &ai_messages,
            model,
            Some(temperature),
            Some(max_tokens as u32),
            stream,
            None,
        )
        .map_err(|e| format!("build_request failed: {:?}", e))?;

    let body = serde_json::to_string(&request.body)
        .map_err(|e| format!("Serialize request failed: {}", e))?;

    Ok((body, stream))
}

/// Parse a non-streaming chat completion response.
pub fn parse_chat_response(response_json: &str) -> Result<(String, String, i64, i64, i64), String> {
    let resp: Value =
        serde_json::from_str(response_json).map_err(|e| format!("Invalid response JSON: {}", e))?;

    let content = resp
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = resp
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("unknown")
        .to_string();

    let (pt, ct, tt) = resp
        .get("usage")
        .map(|u| {
            (
                u.get("prompt_tokens").and_then(|t| t.as_i64()).unwrap_or(0),
                u.get("completion_tokens")
                    .and_then(|t| t.as_i64())
                    .unwrap_or(0),
                u.get("total_tokens").and_then(|t| t.as_i64()).unwrap_or(0),
            )
        })
        .unwrap_or((0, 0, 0));

    Ok((content, finish_reason, pt, ct, tt))
}

/// Parse a single SSE stream event data payload.
pub fn parse_stream_event(data: &str) -> Result<(String, String, bool), String> {
    if is_stream_done(data) {
        return Ok(("stream_end".to_string(), "".to_string(), true));
    }

    let event: Value =
        serde_json::from_str(data).map_err(|e| format!("Invalid stream event JSON: {}", e))?;

    let delta = event
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"));

    let finish_reason = event
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str());

    if let Some(reason) = finish_reason {
        if reason == "stop" || reason == "length" || reason == "content_filter" {
            return Ok(("stream_end".to_string(), reason.to_string(), true));
        }
    }

    if let Some(delta) = delta {
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            return Ok(("content_delta".to_string(), content.to_string(), false));
        }
        if let Some(thinking) = delta.get("reasoning_content").and_then(|t| t.as_str()) {
            return Ok(("thinking_delta".to_string(), thinking.to_string(), false));
        }
        if delta.get("role").is_some() {
            return Ok(("role_assign".to_string(), String::new(), false));
        }
    }

    Ok(("unknown".to_string(), String::new(), false))
}

/// Classify an HTTP error status code using ai-lib-core's StandardErrorCode.
pub fn classify_error(status_code: u16) -> (u32, String, String, bool) {
    let ec = StandardErrorCode::from_http_status(status_code);

    let name = ec.name().to_string();
    let category = ec.category().to_string();
    let retryable = ec.retryable();

    (status_code as u32, name, category, retryable)
}

/// Check if an SSE data payload signals stream completion.
pub fn is_stream_done(data: &str) -> bool {
    data.trim() == "[DONE]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_chat_request_basic() {
        let messages = r#"[{"role":"user","content":"Hello"}]"#;
        let result = build_chat_request(messages, "gpt-4", 0.7, 1024.0, false);
        assert!(result.is_ok());
        let (body, stream) = result.unwrap();
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "gpt-4");
        assert!(!stream);
    }

    #[test]
    fn test_build_chat_request_streaming() {
        let messages = r#"[{"role":"user","content":"Hi"}]"#;
        let result = build_chat_request(messages, "llama-3", 0.5, 2048.0, true);
        assert!(result.is_ok());
        let (_, stream) = result.unwrap();
        assert!(stream);
    }

    #[test]
    fn test_build_chat_request_invalid_json() {
        let result = build_chat_request("not json", "model", 0.7, 100.0, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_chat_request_with_system() {
        let messages =
            r#"[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hi"}]"#;
        let result = build_chat_request(messages, "test-model", 0.7, 100.0, false);
        assert!(result.is_ok());
        let (body, _) = result.unwrap();
        let parsed: Value = serde_json::from_str(&body).unwrap();
        let msgs = parsed["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_parse_chat_response_basic() {
        let response = r#"{
            "choices": [{"message":{"content":"Hello!"},"finish_reason":"stop"}],
            "usage": {"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
        }"#;
        let result = parse_chat_response(response).unwrap();
        assert_eq!(result.0, "Hello!");
        assert_eq!(result.1, "stop");
        assert_eq!(result.2, 5);
        assert_eq!(result.3, 2);
        assert_eq!(result.4, 7);
    }

    #[test]
    fn test_parse_chat_response_empty() {
        let response = r#"{"choices":[]}"#;
        let result = parse_chat_response(response).unwrap();
        assert_eq!(result.0, "");
    }

    #[test]
    fn test_parse_chat_response_invalid() {
        let result = parse_chat_response("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_stream_event_content() {
        let data = r#"{"choices":[{"delta":{"content":"Hi"}}]}"#;
        let result = parse_stream_event(data).unwrap();
        assert_eq!(result.0, "content_delta");
        assert_eq!(result.1, "Hi");
        assert!(!result.2);
    }

    #[test]
    fn test_parse_stream_event_done() {
        let result = parse_stream_event("[DONE]").unwrap();
        assert_eq!(result.0, "stream_end");
        assert!(result.2);
    }

    #[test]
    fn test_parse_stream_event_finish_reason() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let result = parse_stream_event(data).unwrap();
        assert_eq!(result.0, "stream_end");
        assert!(result.2);
    }

    #[test]
    fn test_parse_stream_event_role() {
        let data = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#;
        let result = parse_stream_event(data).unwrap();
        assert_eq!(result.0, "role_assign");
        assert!(!result.2);
    }

    #[test]
    fn test_parse_stream_event_thinking() {
        let data = r#"{"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}"#;
        let result = parse_stream_event(data).unwrap();
        assert_eq!(result.0, "thinking_delta");
        assert_eq!(result.1, "Let me think...");
    }

    #[test]
    fn test_parse_stream_event_invalid() {
        let result = parse_stream_event("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_classify_error_rate_limit() {
        let result = classify_error(429);
        assert_eq!(result.0, 429);
        assert_eq!(result.2, "rate");
        assert!(result.3);
    }

    #[test]
    fn test_classify_error_auth() {
        let result = classify_error(401);
        assert_eq!(result.2, "client");
        assert!(!result.3);
    }

    #[test]
    fn test_classify_error_server() {
        let result = classify_error(500);
        assert_eq!(result.2, "server");
        assert!(result.3);
    }

    #[test]
    fn test_classify_error_not_found() {
        let result = classify_error(404);
        assert_eq!(result.2, "client");
        assert!(!result.3);
    }

    #[test]
    fn test_is_stream_done_true() {
        assert!(is_stream_done("[DONE]"));
        assert!(is_stream_done(" [DONE] "));
    }

    #[test]
    fn test_is_stream_done_false() {
        assert!(!is_stream_done("{\"choices\":[]}"));
        assert!(!is_stream_done("data: [DONE]"));
    }
}
