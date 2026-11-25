/** Hash-based client router mapping routes to views and page headers. */

export const ROUTES = {
  "for-you": {
    title: "For You",
    subtitle: "Personalized novelty-aware news event feed",
  },
  explore: {
    title: "Explore Events",
    subtitle: "Multi-source event clusters and chronological coverage timelines",
  },
  novelty: {
    title: "Novelty Inspector",
    subtitle: "Sentence-level incremental evidence comparison against event memory H(e, t)",
  },
  topics: {
    title: "Emerging Topics",
    subtitle: "Streaming coherent topic cluster detection across recent embeddings",
  },
  interests: {
    title: "My Interests",
    subtitle: "Synthetic demonstration persona profile and engagement logs",
  },
  benchmarks: {
    title: "Benchmarks",
    subtitle: "Offline MIND ranking & corpus novelty saturation evaluations",
  },
};

export function parseRouteFromHash() {
  const raw = (window.location.hash || "#/for-you").replace(/^#\/?/, "");
  const [slug, queryStr] = raw.split("?");
  const route = ROUTES[slug] ? slug : "for-you";
  const params = new URLSearchParams(queryStr || "");
  return { route, params };
}

export function navigateTo(route, params = {}) {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") {
      usp.set(k, String(v));
    }
  }
  const q = usp.toString();
  const nextHash = q ? `#/${route}?${q}` : `#/${route}`;
  if (window.location.hash !== nextHash) {
    window.location.hash = nextHash;
  } else {
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  }
}
