/** Sidebar live news ingestion & clustering control component. */

import { api } from "../api.js";
import { setState, state } from "../state.js";
import { clearNode, el, formatUtcTimestamp } from "../utils/dom.js";
import { showToast } from "./toast.js";

function renderProviderCard(label, p, quotaObj) {
  const isConfigured = Boolean(p?.credentials_configured);
  const statusStr = p?.last_status || (isConfigured ? "READY" : "UNCONFIGURED");
  let badgeCls = "badge badge-subtle";
  if (statusStr === "SUCCESS" || statusStr === "READY") badgeCls = "badge badge-new";
  else if (statusStr === "FAILED" || statusStr === "UNCONFIGURED") badgeCls = "badge badge-repeated";
  else if (statusStr === "EMPTY") badgeCls = "badge badge-warn";

  const rows = [
    el("div", { className: "ingestion-row" }, [
      el("strong", { text: label, style: "color: var(--text-primary);" }),
      el("span", { className: badgeCls, text: statusStr }),
    ]),
    el("div", {
      text: `Sync: ${p?.last_sync_at ? formatUtcTimestamp(p.last_sync_at) : "None yet"}`,
      style: "color: var(--text-faint);",
    }),
    el("div", {
      className: "mono",
      text: `Fetched ${p?.articles_fetched ?? 0} | +${p?.articles_inserted ?? 0} new | ${p?.duplicates_skipped ?? 0} dup`,
      style: "font-size: 11px; color: var(--text-secondary); margin-top: 2px;",
    }),
  ];

  if (quotaObj && quotaObj.daily_remaining !== undefined) {
    rows.push(
      el("div", {
        className: "mono",
        text: `API Quota: ${quotaObj.daily_remaining} / ${quotaObj.daily_limit ?? 5000} left`,
        style: "font-size: 11px; color: var(--accent-primary); margin-top: 2px;",
      })
    );
  }
  if (p?.last_error) {
    rows.push(
      el("div", {
        text: p.last_error,
        style: "font-size: 11px; color: var(--status-rep-text); margin-top: 4px;",
      })
    );
  }

  return el("div", { className: "ingestion-provider-box" }, rows);
}

export async function refreshIngestionStatusAndStats(markSeen = true) {
  try {
    const [statusData, statsData] = await Promise.all([api.getIngestionStatus(), api.getStats()]);
    const patch = { ingestionStatus: statusData, stats: statsData };
    if (markSeen && statusData?.data_version !== undefined) {
      patch.lastSeenDataVersion = statusData.data_version;
      patch.pendingFeedUpdates = false;
    }
    setState(patch);
  } catch (err) {
    showToast(err.message || "Failed to refresh ingestion status", "error");
  }
}

function renderAutoSyncBox(liveSync) {
  const enabled = Boolean(liveSync?.enabled);
  const paused = Boolean(liveSync?.paused);
  const autoClusterEnabled = liveSync?.auto_cluster_enabled !== false;
  const badgeText = !enabled ? "OFF" : paused ? "PAUSED" : "AUTO ON";
  const badgeCls = !enabled ? "badge badge-subtle" : paused ? "badge badge-warn" : "badge badge-new";

  const pending = Number(liveSync?.pending_unassigned_articles ?? 0);
  const minBatch = Number(liveSync?.auto_cluster_min_batch ?? 10);
  const rssSec = Number(liveSync?.rss_interval_seconds ?? 300);

  const toggleBtn = el("button", {
    type: "button",
    className: "btn btn-sm",
    disabled: !enabled,
    text: paused ? "Resume Auto" : "Pause Auto",
    onClick: async () => {
      try {
        const updated = await api.updateAutoSync({ paused: !paused });
        const curStatus = state.ingestionStatus || {};
        setState({ ingestionStatus: { ...curStatus, live_sync: updated } });
        showToast(
          updated.paused ? "Background auto-sync paused." : "Background auto-sync resumed.",
          "info"
        );
      } catch (err) {
        showToast(err.message || "Failed to update auto-sync", "error");
      }
    },
  });

  const intervalSelect = el(
    "select",
    {
      className: "select-input autosync-select",
      disabled: !enabled,
      ariaLabel: "RSS auto-sync interval",
      onChange: async (e) => {
        const nextSec = Number(e.target.value);
        try {
          const updated = await api.updateAutoSync({ rss_interval_seconds: nextSec });
          const curStatus = state.ingestionStatus || {};
          setState({ ingestionStatus: { ...curStatus, live_sync: updated } });
          showToast(`RSS auto-sync interval set to ${Math.round(nextSec / 60)} min.`, "info");
        } catch (err) {
          showToast(err.message || "Failed to update sync interval", "error");
        }
      },
    },
    [
      [120, "2 min"],
      [300, "5 min"],
      [900, "15 min"],
      [1800, "30 min"],
    ].map(([sec, lbl]) =>
      el("option", { value: String(sec), text: lbl, selected: rssSec === sec })
    )
  );

  const clusterToggleBtn = el("button", {
    type: "button",
    className: "btn btn-sm",
    text: autoClusterEnabled ? "Auto-Cluster: ON" : "Auto-Cluster: OFF",
    onClick: async () => {
      try {
        const updated = await api.updateAutoSync({
          auto_cluster_enabled: !autoClusterEnabled,
        });
        const curStatus = state.ingestionStatus || {};
        setState({ ingestionStatus: { ...curStatus, live_sync: updated } });
        showToast(
          updated.auto_cluster_enabled
            ? `Batch auto-clustering enabled (>= ${updated.auto_cluster_min_batch} articles).`
            : "Batch auto-clustering turned off.",
          "info"
        );
      } catch (err) {
        showToast(err.message || "Failed to toggle auto-clustering", "error");
      }
    },
  });

  const batchSelect = el(
    "select",
    {
      className: "select-input autosync-select",
      disabled: !autoClusterEnabled,
      ariaLabel: "Auto-cluster batch threshold",
      onChange: async (e) => {
        const nextBatch = Number(e.target.value);
        try {
          const updated = await api.updateAutoSync({ auto_cluster_min_batch: nextBatch });
          const curStatus = state.ingestionStatus || {};
          setState({ ingestionStatus: { ...curStatus, live_sync: updated } });
          showToast(`Auto-cluster batch threshold set to ${nextBatch} articles.`, "info");
        } catch (err) {
          showToast(err.message || "Failed to update cluster batch size", "error");
        }
      },
    },
    [
      [5, "Batch 5"],
      [10, "Batch 10"],
      [20, "Batch 20"],
      [50, "Batch 50"],
    ].map(([cnt, lbl]) =>
      el("option", { value: String(cnt), text: lbl, selected: minBatch === cnt })
    )
  );

  return el("div", { className: "ingestion-provider-box" }, [
    el("div", { className: "ingestion-row" }, [
      el("strong", { text: "Background Auto-Sync", style: "color: var(--text-primary);" }),
      el("span", { className: badgeCls, text: badgeText }),
    ]),
    el("div", {
      text: liveSync?.next_rss_sync_at
        ? `Next RSS: ${formatUtcTimestamp(liveSync.next_rss_sync_at)}`
        : paused
        ? "Schedule paused"
        : "Schedule inactive",
      style: "color: var(--text-faint);",
    }),
    el("div", {
      className: "mono",
      text: autoClusterEnabled
        ? `Cluster queue: ${pending} / ${minBatch} articles`
        : `Cluster queue: ${pending} (auto off)`,
      style: "font-size: 11px; color: var(--text-secondary); margin-top: 2px;",
    }),
    el("div", { className: "autosync-controls" }, [toggleBtn, intervalSelect]),
    el("div", { className: "autosync-controls" }, [clusterToggleBtn, batchSelect]),
  ]);
}

export function renderIngestionPanel(host, onDataChanged) {
  if (!host) return;
  clearNode(host);

  const st = state.ingestionStatus;
  const busy =
    state.localJobBusy ||
    Boolean(st?.ingestion_job_running) ||
    Boolean(st?.clustering_job_running);

  host.appendChild(
    el("div", { className: "field-label" }, [
      el("span", { text: "Live News Streams" }),
      busy ? el("span", { className: "badge badge-warn", text: "RUNNING" }) : null,
    ])
  );

  host.appendChild(
    renderProviderCard("FreeNewsAPI (IN/en)", st?.freenewsapi, st?.freenewsapi?.quota)
  );
  host.appendChild(renderProviderCard("RSS Feeds (23)", st?.rss, null));
  host.appendChild(renderAutoSyncBox(st?.live_sync));

  const runFnBtn = el("button", {
    type: "button",
    className: "btn btn-primary btn-sm",
    disabled: busy,
    text: busy ? "Syncing..." : "Refresh FreeNewsAPI",
    onClick: async () => {
      setState({ localJobBusy: true });
      try {
        const res = await api.runFreeNewsApiIngestion({
          country: "IN",
          language: "en",
          max_pages_per_topic: 1,
          max_details_to_fetch: 10,
        });
        showToast(
          `FreeNewsAPI: ${res.articles_fetched} listed, +${res.articles_inserted} inserted, ${res.duplicates_skipped} duplicates skipped.`,
          res.articles_inserted > 0 ? "success" : "info"
        );
        await refreshIngestionStatusAndStats();
        if (onDataChanged) onDataChanged();
      } catch (err) {
        showToast(err.message || "FreeNewsAPI ingestion failed", "error");
        await refreshIngestionStatusAndStats();
      } finally {
        setState({ localJobBusy: false });
      }
    },
  });

  const runRssBtn = el("button", {
    type: "button",
    className: "btn btn-sm",
    disabled: busy,
    text: busy ? "Syncing..." : "Refresh RSS Feeds",
    onClick: async () => {
      setState({ localJobBusy: true });
      try {
        const res = await api.runRssIngestion({});
        showToast(
          `RSS (${res.status}): ${res.articles_fetched} fetched, +${res.articles_inserted} inserted, ${res.duplicates_skipped} duplicates.`,
          res.articles_inserted > 0 ? "success" : "info"
        );
        await refreshIngestionStatusAndStats();
        if (onDataChanged) onDataChanged();
      } catch (err) {
        showToast(err.message || "RSS ingestion failed", "error");
        await refreshIngestionStatusAndStats();
      } finally {
        setState({ localJobBusy: false });
      }
    },
  });

  const clusterBtn = el("button", {
    type: "button",
    className: "btn btn-sm",
    disabled: busy,
    text: busy ? "Clustering..." : "Cluster New Articles",
    onClick: async () => {
      setState({ localJobBusy: true });
      try {
        const res = await api.runClustering({ include_sample: false });
        showToast(
          `Clustering complete: ${res.articles_clustered} articles assigned (${res.new_events_created} new events).`,
          "success"
        );
        await refreshIngestionStatusAndStats();
        if (onDataChanged) onDataChanged();
      } catch (err) {
        showToast(err.message || "Clustering execution failed", "error");
      } finally {
        setState({ localJobBusy: false });
      }
    },
  });

  host.appendChild(
    el("div", { className: "ingestion-actions" }, [runFnBtn, runRssBtn, clusterBtn])
  );
}
