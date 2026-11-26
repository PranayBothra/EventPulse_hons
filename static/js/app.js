/** Main ES module entry point bootstrapping EventPulse SPA state, navigation, and views. */

import { api } from "./api.js";
import { renderIngestionPanel, refreshIngestionStatusAndStats } from "./components/ingestionPanel.js";
import { showToast } from "./components/toast.js";
import { parseRouteFromHash, ROUTES } from "./router.js";
import { setState, state, subscribe } from "./state.js";
import { clearNode, el } from "./utils/dom.js";
import { renderBenchmarksView } from "./views/benchmarksView.js";
import { renderEmergingTopicsView } from "./views/emergingTopicsView.js";
import { renderExploreEventsView } from "./views/exploreEventsView.js";
import { renderForYouView } from "./views/forYouView.js";
import { renderMyInterestsView } from "./views/myInterestsView.js";
import { renderNoveltyInspectorView } from "./views/noveltyInspectorView.js";

const viewRoot = document.getElementById("view-root");
const liveBanner = document.getElementById("live-update-banner");
const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const telemetryHost = document.getElementById("topbar-telemetry");
const personaSelect = document.getElementById("global-persona-select");
const personaMeta = document.getElementById("global-persona-meta");
const ingestionHost = document.getElementById("ingestion-sidebar-panel");
const sidebar = document.getElementById("sidebar");
const backdrop = document.getElementById("sidebar-backdrop");
const mobileMenuBtn = document.getElementById("mobile-menu-btn");

function setMobileDrawerOpen(open) {
  sidebar.classList.toggle("open", open);
  backdrop.classList.toggle("visible", open);
}

mobileMenuBtn?.addEventListener("click", () => setMobileDrawerOpen(!sidebar.classList.contains("open")));
backdrop?.addEventListener("click", () => setMobileDrawerOpen(false));

function renderTopTelemetry() {
  if (!telemetryHost) return;
  clearNode(telemetryHost);
  const st = state.stats;
  if (!st) return;

  const fnCount = st.sources
    ? Object.entries(st.sources)
        .filter(([k]) => k.startsWith("freenewsapi_"))
        .reduce((acc, [, v]) => acc + Number(v || 0), 0)
    : 0;

  const items = [
    [Number(st.total_articles || 0).toLocaleString(), "Articles"],
    [Number(st.total_events || 0).toLocaleString(), "Events"],
    [String(st.distinct_sources_count || 0), "Sources"],
    [String(fnCount), "Live FreeNewsAPI"],
  ];

  for (const [val, label] of items) {
    telemetryHost.appendChild(
      el("div", { className: "telemetry-item" }, [
        el("span", { className: "telemetry-value", text: val }),
        el("span", { className: "telemetry-label", text: label }),
      ])
    );
  }
}

function updatePersonaControls() {
  if (!personaSelect) return;
  clearNode(personaSelect);
  for (const p of state.presets) {
    personaSelect.appendChild(
      el("option", {
        value: p.user_id,
        text: p.display_name,
        selected: p.user_id === state.selectedUserId,
      })
    );
  }
  const active = state.presets.find((p) => p.user_id === state.selectedUserId);
  if (personaMeta) {
    personaMeta.textContent = active ? active.description : "";
  }
}

personaSelect?.addEventListener("change", (e) => {
  setState({ selectedUserId: e.target.value });
  updatePersonaControls();
  renderCurrentRoute();
});

function updateActiveNav(route) {
  const links = document.querySelectorAll("#nav-list .nav-item");
  links.forEach((lnk) => {
    lnk.classList.toggle("active", lnk.dataset.route === route);
  });
}

async function renderCurrentRoute() {
  const { route, params } = parseRouteFromHash();
  setState({ activeRoute: route, pendingFeedUpdates: false });
  updateActiveNav(route);
  setMobileDrawerOpen(false);

  const meta = ROUTES[route] || ROUTES["for-you"];
  pageTitle.textContent = meta.title;
  pageSubtitle.textContent = meta.subtitle;

  if (route === "for-you") {
    await renderForYouView(viewRoot, params, () => refreshIngestionStatusAndStats());
  } else if (route === "explore") {
    await renderExploreEventsView(viewRoot, params);
  } else if (route === "novelty") {
    await renderNoveltyInspectorView(viewRoot, params);
  } else if (route === "topics") {
    await renderEmergingTopicsView(viewRoot);
  } else if (route === "interests") {
    await renderMyInterestsView(viewRoot);
  } else if (route === "benchmarks") {
    await renderBenchmarksView(viewRoot);
  }
}

subscribe((_st, patch) => {
  if (patch.stats !== undefined) {
    renderTopTelemetry();
  }
  if (patch.ingestionStatus !== undefined || patch.localJobBusy !== undefined) {
    renderIngestionPanel(ingestionHost, () => renderCurrentRoute());
  }
  if (patch.pendingFeedUpdates !== undefined) {
    renderLiveUpdateBanner();
  }
});

function renderLiveUpdateBanner() {
  if (!liveBanner) return;
  clearNode(liveBanner);
  if (!state.pendingFeedUpdates) {
    liveBanner.classList.add("hidden");
    return;
  }

  liveBanner.classList.remove("hidden");
  const pendingUnassigned = state.ingestionStatus?.live_sync?.pending_unassigned_articles ?? 0;
  const msg =
    pendingUnassigned > 0
      ? `New live articles synced in the background (${pendingUnassigned} awaiting batch clustering).`
      : "New live stories and event updates are available.";

  liveBanner.appendChild(el("span", { text: msg }));
  liveBanner.appendChild(
    el("div", { className: "live-update-actions" }, [
      el("button", {
        type: "button",
        className: "btn btn-primary btn-sm",
        text: "Refresh Feed",
        onClick: async () => {
          setState({
            pendingFeedUpdates: false,
            lastSeenDataVersion: state.ingestionStatus?.data_version ?? state.lastSeenDataVersion,
          });
          await renderCurrentRoute();
        },
      }),
      el("button", {
        type: "button",
        className: "btn btn-sm",
        text: "Dismiss",
        onClick: () => {
          setState({
            pendingFeedUpdates: false,
            lastSeenDataVersion: state.ingestionStatus?.data_version ?? state.lastSeenDataVersion,
          });
        },
      }),
    ])
  );
}

let lastPollAt = 0;
let pollInFlight = false;

async function pollLiveStatus(force = false) {
  if (pollInFlight) return;
  if (!force && document.visibilityState !== "visible") return;
  const now = Date.now();
  if (!force && now - lastPollAt < 5000) return;

  pollInFlight = true;
  lastPollAt = now;
  try {
    const ingStatus = await api.getIngestionStatus();
    const serverVer = Number(ingStatus?.data_version ?? 1);
    const seenVer = state.lastSeenDataVersion;

    if (seenVer === null) {
      setState({ ingestionStatus: ingStatus, lastSeenDataVersion: serverVer });
      return;
    }

    if (serverVer > seenVer) {
      const statsData = await api.getStats();
      if (state.autoApplyFeedUpdates) {
        setState({
          ingestionStatus: ingStatus,
          stats: statsData,
          lastSeenDataVersion: serverVer,
          pendingFeedUpdates: false,
        });
        await renderCurrentRoute();
      } else {
        setState({
          ingestionStatus: ingStatus,
          stats: statsData,
          lastSeenDataVersion: serverVer,
          pendingFeedUpdates: true,
        });
      }
      showToast("Background sync updated live news data.", "info");
    } else {
      // Only update sidebar status cleanly without touching active view
      setState({ ingestionStatus: ingStatus });
    }
  } catch {
    // Silent retry on transient polling failure
  } finally {
    pollInFlight = false;
  }
}

async function bootstrap() {
  window.addEventListener("hashchange", () => {
    renderCurrentRoute();
  });

  renderIngestionPanel(ingestionHost, () => renderCurrentRoute());

  try {
    const [presets, stats, ingStatus] = await Promise.all([
      api.getUserPresets(),
      api.getStats(),
      api.getIngestionStatus(),
    ]);
    const validPreset = presets.some((p) => p.user_id === state.selectedUserId);
    setState({
      presets,
      stats,
      ingestionStatus: ingStatus,
      lastSeenDataVersion: Number(ingStatus?.data_version ?? 1),
      selectedUserId: validPreset ? state.selectedUserId : presets[0]?.user_id || "user_tech",
    });
    updatePersonaControls();
  } catch (err) {
    showToast(err.message || "Failed to load initial EventPulse metadata.", "error");
  }

  await renderCurrentRoute();

  // Lightweight visibility-aware polling (every 20s while tab is visible)
  setInterval(() => {
    pollLiveStatus(false);
  }, 20000);

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      pollLiveStatus(false);
    }
  });
}

bootstrap();
