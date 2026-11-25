/** View D: Emerging Topics — Streaming Topic Emergence Detector Output. */

import { api } from "../api.js";
import { navigateTo } from "../router.js";
import { setState, state } from "../state.js";
import { renderCardSkeletons, renderEmptyState, renderErrorState } from "../components/skeletons.js";
import { clearNode, el, formatUtcTimestamp } from "../utils/dom.js";

export async function renderEmergingTopicsView(container) {
  clearNode(container);

  let articleWindow = 400;
  let minClusterSize = 2;

  const detectorSelect = el(
    "select",
    {
      className: "select-input",
      onChange: (e) => {
        setState({ topicsDetector: e.target.value });
        loadTopics();
      },
    },
    [
      el("option", {
        value: "online_cluster",
        text: "EventPulseClusterEmergenceTracker",
        selected: state.topicsDetector === "online_cluster",
      }),
      el("option", {
        value: "rolling_window",
        text: "RollingWindowEmergenceDetector",
        selected: state.topicsDetector === "rolling_window",
      }),
    ]
  );

  const windowSelect = el(
    "select",
    {
      className: "select-input",
      onChange: (e) => {
        articleWindow = Number(e.target.value);
        loadTopics();
      },
    },
    [200, 400, 800, 1200].map((w) =>
      el("option", { value: String(w), text: `Last ${w} articles`, selected: w === articleWindow })
    )
  );

  const minSizeSelect = el(
    "select",
    {
      className: "select-input",
      onChange: (e) => {
        minClusterSize = Number(e.target.value);
        loadTopics();
      },
    },
    [2, 3, 5].map((m) =>
      el("option", { value: String(m), text: `Min ${m} articles`, selected: m === minClusterSize })
    )
  );

  container.appendChild(
    el("div", { className: "toolbar" }, [
      el("div", { className: "toolbar-group grow" }, [
        el("span", { className: "field-label", text: "Emergence Detector (src/novelty/emerging_topic.py)" }),
        detectorSelect,
      ]),
      el("div", { className: "toolbar-group" }, [
        el("span", { className: "field-label", text: "Stream Window" }),
        windowSelect,
      ]),
      el("div", { className: "toolbar-group" }, [
        el("span", { className: "field-label", text: "Coherence Threshold" }),
        minSizeSelect,
      ]),
    ])
  );

  const metaBanner = el("div", { className: "notice-banner" });
  const topicsHost = el("div");
  container.appendChild(metaBanner);
  container.appendChild(topicsHost);

  async function loadTopics() {
    clearNode(topicsHost);
    topicsHost.appendChild(renderCardSkeletons(3));
    metaBanner.textContent = "Scanning cached article embeddings for emerging topic clusters...";

    try {
      const res = await api.getTopics({
        detector: state.topicsDetector,
        article_window: articleWindow,
        min_cluster_size: minClusterSize,
        limit: 30,
      });

      metaBanner.textContent = `Detector: ${res.detector_used} · Articles scanned: ${res.articles_scanned} · Coherent topics detected: ${res.topics_found}.`;
      clearNode(topicsHost);

      if (!res.topics || !res.topics.length) {
        topicsHost.appendChild(
          renderEmptyState(
            "No Emergent Clusters Found",
            "Try increasing the stream window or lowering the minimum cluster size."
          )
        );
        return;
      }

      const grid = el("div", { className: "grid-cards" });
      for (const t of res.topics) {
        const onsetBadge = el("span", {
          className: t.seeded_before_onset ? "badge badge-subtle" : "badge badge-accent",
          text: t.seeded_before_onset ? "Pre-Midpoint Seeded" : "Post-Midpoint Emergence",
        });

        grid.appendChild(
          el("article", { className: "story-card", style: "margin-bottom: 0;" }, [
            el("div", { className: "story-meta-row" }, [
              el("span", { className: "mono story-rank", text: `Cluster #${t.cluster_id}` }),
              onsetBadge,
            ]),
            el("h3", {
              className: "story-headline",
              text: t.representative_headline,
              style: "font-size: 17px;",
            }),
            el("div", { className: "score-strip", style: "margin-bottom: 10px;" }, [
              el("div", { className: "score-cell" }, [
                el("div", { className: "score-cell-label", text: "Volume" }),
                el("div", { className: "score-cell-value", text: `${t.article_count} arts` }),
              ]),
              el("div", { className: "score-cell" }, [
                el("div", { className: "score-cell-label", text: "Stream Span" }),
                el("div", { className: "score-cell-value", text: `${t.first_step} -> ${t.last_step}` }),
              ]),
            ]),
            el("div", {
              className: "mono",
              text: `Latest: ${formatUtcTimestamp(t.latest_published_at)}`,
              style: "font-size: 11.5px; color: var(--text-faint); margin-bottom: 8px;",
            }),
            el("div", {
              className: "action-group",
              style: "gap: 5px; margin-bottom: 10px;",
            }, (t.sources || []).slice(0, 5).map((src) => el("span", { className: "badge badge-subtle", text: src }))),
            el("div", { className: "story-actions" }, [
              el("span", { className: "mono", text: `Rep Article #${t.representative_article_id}`, style: "font-size: 11px; color: var(--text-faint);" }),
              el("button", {
                type: "button",
                className: "btn btn-sm",
                text: "Browse Events",
                onClick: () => navigateTo("explore"),
              }),
            ]),
          ])
        );
      }
      topicsHost.appendChild(grid);
    } catch (err) {
      clearNode(topicsHost);
      topicsHost.appendChild(
        renderErrorState(err.message || "Failed to detect emerging topics.", loadTopics)
      );
    }
  }

  await loadTopics();
}
