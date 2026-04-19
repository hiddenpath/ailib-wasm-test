//! # ai-wasm-test-server
//!
//! 后端代理服务 — 将浏览器 WASM 构建的请求转发给 AI provider。
//!
//! Backend proxy server forwarding browser-WASM-built requests to AI providers.
//! The browser wasm module handles request building and response parsing;
//! this server only proxies HTTP to avoid CORS and protect API keys.
//!
//! Uses libcurl (via the `curl` crate) for outbound requests because some
//! AI providers (notably Groq) block Rust's native TLS fingerprints (JA3/JA4).
//! libcurl shares the same TLS fingerprint as the system `curl` binary.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use curl::easy::Easy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {}

/// Resolve the API key for a provider based on the request URL.
fn resolve_api_key(url: &str) -> Option<String> {
    if url.contains("api.groq.com") {
        return std::env::var("GROQ_API_KEY").ok();
    }
    if url.contains("api.deepseek.com") {
        return std::env::var("DEEPSEEK_API_KEY").ok();
    }
    if url.contains("api.openai.com") {
        return std::env::var("OPENAI_API_KEY").ok();
    }
    None
}

/// Incoming proxy request from the browser wasm module.
#[derive(Deserialize)]
pub struct ProxyRequest {
    pub url: String,
    #[serde(default)]
    pub headers: serde_json::Map<String, Value>,
    pub body: Value,
    pub stream: bool,
}

/// Non-streaming proxy response returned to the browser.
#[derive(Serialize)]
pub struct ProxyResponse {
    pub status: u16,
    pub body: Value,
}

/// Health check response.
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Resolve the static directory path.
/// Tries CWD-relative first, then executable-relative, then hard-coded fallback.
fn static_dir() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    let candidates = [
        cwd.join("static"),
        std::path::PathBuf::from("/home/alex/ai-wasm-test/static"),
    ];
    for c in &candidates {
        if c.join("index.html").exists() {
            return c.clone();
        }
    }
    cwd.join("static")
}

/// Serve index.html for the root path.
async fn index_handler() -> Response {
    let path = static_dir().join("index.html");
    match std::fs::read_to_string(&path) {
        Ok(html) => axum::response::Html(html).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "index.html not found").into_response(),
    }
}

/// Build the axum Router with all routes and middleware.
pub fn create_app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_handler))
        .route("/", get(index_handler))
        .route("/api/proxy", post(proxy_handler))
        .route("/api/proxy/stream", post(proxy_stream_handler))
        .fallback_service(tower_http::services::ServeDir::new(static_dir()))
        .layer(cors)
        .with_state(Arc::new(state))
}

/// GET /health — returns server status and version.
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Execute a non-streaming proxy request using libcurl.
fn curl_proxy(url: &str, headers: &serde_json::Map<String, Value>, body: &Value) -> ProxyResponse {
    let mut easy = Easy::new();
    easy.url(url).unwrap();
    easy.post(true).unwrap();
    easy.timeout(Duration::from_secs(120)).unwrap();

    let body_str = serde_json::to_string(body).unwrap_or_default();
    easy.post_fields_copy(body_str.as_bytes()).unwrap();

    let mut list = curl::easy::List::new();
    list.append("Content-Type: application/json").unwrap();

    if let Some(key) = resolve_api_key(url) {
        list.append(&format!("Authorization: Bearer {}", key))
            .unwrap();
    }

    for (key, val) in headers {
        if key.eq_ignore_ascii_case("authorization") || key.eq_ignore_ascii_case("content-type") {
            continue;
        }
        if let Some(s) = val.as_str() {
            list.append(&format!("{}: {}", key, s)).unwrap();
        }
    }
    easy.http_headers(list).unwrap();

    let mut resp_data = Vec::new();
    {
        let mut transfer = easy.transfer();
        transfer
            .write_function(|data| {
                resp_data.extend_from_slice(data);
                Ok(data.len())
            })
            .unwrap();
        transfer.perform().unwrap();
    }
    let status_code = easy.response_code().unwrap() as u16;

    let resp_body: Value = serde_json::from_slice(&resp_data).unwrap_or(Value::Null);
    ProxyResponse {
        status: status_code,
        body: resp_body,
    }
}

/// POST /api/proxy — non-streaming proxy: forward request, return full response.
async fn proxy_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ProxyRequest>,
) -> impl IntoResponse {
    let url = req.url.clone();
    let headers = req.headers.clone();
    let body = req.body.clone();

    let result = tokio::task::spawn_blocking(move || curl_proxy(&url, &headers, &body))
        .await
        .unwrap_or(ProxyResponse {
            status: 502,
            body: serde_json::json!({"error": "blocking task failed"}),
        });

    let code = StatusCode::from_u16(result.status).unwrap_or(StatusCode::BAD_GATEWAY);
    (code, Json(result))
}

/// POST /api/proxy/stream — SSE streaming proxy via curl subprocess.
async fn proxy_stream_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ProxyRequest>,
) -> Response {
    let url = req.url.clone();
    let body_str = serde_json::to_string(&req.body).unwrap_or_default();

    let auth_header = match resolve_api_key(&url) {
        Some(key) => format!("Authorization: Bearer {}", key),
        None => String::new(),
    };

    let mut cmd = tokio::process::Command::new("curl");
    cmd.arg("-s")
        .arg("-N")
        .arg("-X")
        .arg("POST")
        .arg(&url)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("--data-binary")
        .arg(&body_str)
        .arg("--max-time")
        .arg("120")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());

    if !auth_header.is_empty() {
        cmd.arg("-H").arg(&auth_header);
    }

    match cmd.spawn() {
        Ok(child) => {
            let stdout = child.stdout.expect("stdout piped");
            use tokio::io::{AsyncBufReadExt, BufReader};
            let reader = BufReader::new(stdout);

            let stream = tokio_stream::wrappers::LinesStream::new(reader.lines());
            let sse_stream = stream.filter_map(|line_result| match line_result {
                Ok(line) => {
                    let stripped = line.strip_prefix("data: ").unwrap_or(&line).to_string();
                    if stripped.is_empty() {
                        None
                    } else {
                        Some(Ok::<_, Infallible>(
                            axum::response::sse::Event::default().data(stripped),
                        ))
                    }
                }
                Err(_) => None,
            });

            Sse::new(sse_stream)
                .keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(15)))
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Failed to spawn curl: {}", e),
        )
            .into_response(),
    }
}

/// Start the server on port 3000.
pub async fn run_server() -> anyhow::Result<()> {
    let state = AppState {};
    let app = create_app(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("ai-wasm-test-server running on http://localhost:3000");
    axum::serve(listener, app).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run_server().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode as AxumStatusCode};
    use tower::ServiceExt;

    fn test_app() -> Router {
        let state = AppState {};
        create_app(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = test_app();
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), AxumStatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let health: HealthResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(health.status, "ok");
        assert_eq!(health.version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_proxy_missing_url_returns_error() {
        let app = test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/proxy")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"headers":{},"body":{},"stream":false}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), AxumStatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let app = test_app();
        let req = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), AxumStatusCode::NOT_FOUND);
    }

    #[test]
    fn test_resolve_api_key_groq() {
        let key = resolve_api_key("https://api.groq.com/openai/v1/chat/completions");
        assert!(key.is_some(), "GROQ_API_KEY should be set");
    }

    #[test]
    fn test_resolve_api_key_deepseek() {
        let key = resolve_api_key("https://api.deepseek.com/chat/completions");
        assert!(key.is_some(), "DEEPSEEK_API_KEY should be set");
    }

    #[test]
    fn test_resolve_api_key_unknown() {
        let key = resolve_api_key("https://unknown.provider.com/v1/chat");
        assert!(key.is_none(), "Unknown provider should return None");
    }

    #[test]
    fn test_curl_proxy_groq() {
        let key = std::env::var("GROQ_API_KEY");
        if key.is_err() {
            eprintln!("Skipping: GROQ_API_KEY not set");
            return;
        }
        let result = curl_proxy(
            "https://api.groq.com/openai/v1/chat/completions",
            &serde_json::Map::new(),
            &serde_json::json!({
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": false
            }),
        );
        assert_eq!(
            result.status, 200,
            "Groq proxy should return 200, got {} body: {:?}",
            result.status, result.body
        );
    }

    #[test]
    fn test_curl_proxy_deepseek() {
        let key = std::env::var("DEEPSEEK_API_KEY");
        if key.is_err() {
            eprintln!("Skipping: DEEPSEEK_API_KEY not set");
            return;
        }
        let result = curl_proxy(
            "https://api.deepseek.com/chat/completions",
            &serde_json::Map::new(),
            &serde_json::json!({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": false
            }),
        );
        assert_eq!(
            result.status, 200,
            "DeepSeek proxy should return 200, got {} body: {:?}",
            result.status, result.body
        );
    }
}
