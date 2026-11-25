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

export async function refreshIngestionStatusAndStats() {
  try {
    const [statusData, statsData] = await Promise.all([api.getIngestionStatus(), api.getStats()]);
    setState({ ingestionStatus: statusData, stats: statsData });
  } catch (err) {
    showToast(err.message || "Failed to refresh ingestion status", "error");
  }
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
