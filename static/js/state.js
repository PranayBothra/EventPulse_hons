/** Centralized client-side state store with lightweight subscription notifications. */

export const DEFAULT_WEIGHTS = {
  w_long: 0.40,
  w_short: 0.35,
  w_event_novelty: 0.05,
  w_user_novelty: 0.05,
  w_freshness: 0.10,
  w_redundancy_penalty: 0.05,
  mmr_lambda: 0.95,
};

const listeners = new Set();

export const state = {
  activeRoute: "for-you",
  selectedUserId: localStorage.getItem("ep_selected_user") || "user_tech",
  presets: [],
  stats: null,

  // For You settings
  algorithm: "EventPulse",
  mode: "RELEVANCE_FOCUSED",
  topK: 10,
  useCustomWeights: false,
  weights: { ...DEFAULT_WEIGHTS },
  showWeightsDrawer: false,

  // Explore Events selection
  selectedEventId: null,
  eventSearch: "",
  eventStatusFilter: "",
  eventSortBy: "article_count",

  // Novelty Inspector selection
  noveltyEventId: null,
  noveltyArticleId: null,

  // Emerging Topics config
  topicsDetector: "online_cluster",

  // Ingestion & Clustering status
  ingestionStatus: null,
  localJobBusy: false,
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function setState(patch) {
  Object.assign(state, patch);
  if (patch.selectedUserId) {
    try {
      localStorage.setItem("ep_selected_user", patch.selectedUserId);
    } catch {
      // ignore storage quota errors
    }
  }
  for (const fn of listeners) {
    fn(state, patch);
  }
}
