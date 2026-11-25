/** Centralized Fetch API client for EventPulse FastAPI backend. */

export class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function buildQuery(params = {}) {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") {
      usp.set(k, String(v));
    }
  }
  const qs = usp.toString();
  return qs ? `?${qs}` : "";
}

async function request(path, options = {}) {
  const { method = "GET", params, body } = options;
  const url = `${path}${buildQuery(params)}`;
  const init = {
    method,
    headers: { Accept: "application/json" },
  };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  let res;
  try {
    res = await fetch(url, init);
  } catch (err) {
    throw new ApiError("Network error communicating with EventPulse API.", 0, err?.message);
  }

  let data = null;
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      data = await res.json();
    } catch {
      data = null;
    }
  }

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.error || data.message)) ||
      `Request failed with HTTP ${res.status}`;
    throw new ApiError(String(msg), res.status, data);
  }
  return data;
}

export const api = {
  getHealth: () => request("/api/health"),
  getStats: () => request("/api/stats"),

  getUserPresets: () => request("/api/users/presets"),
  getUserProfile: (userId) => request(`/api/users/${encodeURIComponent(userId)}/profile`),
  getUserInteractions: (userId, limit = 25) =>
    request(`/api/users/${encodeURIComponent(userId)}/interactions`, { params: { limit } }),

  getRecommendations: (params) => request("/api/recommendations", { params }),
  postInteraction: (payload) => request("/api/interactions", { method: "POST", body: payload }),

  getEvents: (params) => request("/api/events", { params }),
  getEventArticles: (eventId) => request(`/api/events/${encodeURIComponent(eventId)}/articles`),
  getTopics: (params) => request("/api/topics", { params }),

  getEvolvingEvents: (params) => request("/api/novelty/evolving-events", { params }),
  inspectNovelty: (params) => request("/api/novelty/inspect", { params }),

  getMindBenchmark: () => request("/api/benchmarks/mind"),
  getCorpusNoveltyBenchmark: () => request("/api/benchmarks/corpus-novelty"),

  getIngestionStatus: () => request("/api/ingestion/status"),
  getIngestionHistory: (limit = 10) => request("/api/ingestion/history", { params: { limit } }),
  runFreeNewsApiIngestion: (payload = {}) =>
    request("/api/ingestion/freenewsapi/run", { method: "POST", body: payload }),
  runRssIngestion: (params = {}) => request("/api/ingestion/rss/run", { method: "POST", params }),
  runUnifiedIngestion: (params = {}) => request("/api/ingestion/run", { method: "POST", params }),
  runClustering: (params = {}) => request("/api/clustering/run", { method: "POST", params }),
};
