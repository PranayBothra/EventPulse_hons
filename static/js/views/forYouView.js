/** View A: For You — Personalized Novelty-Aware Event Feed. */

import { api } from "../api.js";
import { navigateTo } from "../router.js";
import { setState, state } from "../state.js";
import { renderCardSkeletons, renderEmptyState, renderErrorState } from "../components/skeletons.js";
import { showToast } from "../components/toast.js";
import { clearNode, el, formatUtcTimestamp, safeExternalUrl } from "../utils/dom.js";

const MODE_PRESET_WEIGHTS = {
  RELEVANCE_FOCUSED: {
    w_long: 0.40,
    w_short: 0.35,
    w_event_novelty: 0.05,
    w_user_novelty: 0.05,
    w_freshness: 0.10,
    w_redundancy_penalty: 0.05,
    mmr_lambda: 0.95,
  },
  BALANCED_INTELLIGENCE: {
    w_long: 0.25,
    w_short: 0.25,
    w_event_novelty: 0.15,
    w_user_novelty: 0.10,
    w_freshness: 0.10,
    w_redundancy_penalty: 0.10,
    mmr_lambda: 0.80,
  },
  DISCOVERY_FOCUSED: {
    w_long: 0.15,
    w_short: 0.15,
    w_event_novelty: 0.25,
    w_user_novelty: 0.20,
    w_freshness: 0.10,
    w_redundancy_penalty: 0.20,
    mmr_lambda: 0.65,
  },
};

const WEIGHT_LABELS = [
  ["w_long", "Long-Term Interest (w1)"],
  ["w_short", "Short-Term Interest (w2)"],
  ["w_event_novelty", "Event Novelty (w3)"],
  ["w_user_novelty", "User Novelty (w4)"],
  ["w_freshness", "Freshness Decay (w5)"],
  ["w_redundancy_penalty", "Redundancy Penalty (w8)"],
  ["mmr_lambda", "MMR Diversity (lambda)"],
];

function makeScoreCells(rec, algorithm) {
  const bd = rec.score_breakdown || {};
  const cells = [
    ["Composite Score", Number(rec.score ?? 0).toFixed(4)],
  ];
  if (algorithm === "Baseline_A") {
    cells.push(["Recency Rank", Number(bd.recency_rank ?? 1).toFixed(2)]);
  } else if (algorithm === "Baseline_B") {
    cells.push(["Content Similarity", Number(bd.content_similarity ?? rec.score ?? 0).toFixed(4)]);
  } else {
    cells.push(["Long-Term Rel", Number(bd.long_term_relevance ?? 0).toFixed(2)]);
    cells.push(["Short-Term Rel", Number(bd.short_term_relevance ?? 0).toFixed(2)]);
    cells.push(["Event Novelty", Number(bd.event_novelty ?? 0).toFixed(2)]);
    cells.push(["User Novelty", Number(bd.user_novelty ?? 0).toFixed(2)]);
    cells.push(["Redundancy", `-${Number(bd.redundancy_penalty ?? 0).toFixed(2)}`]);
  }
  return cells.map(([label, val]) =>
    el("div", { className: "score-cell" }, [
      el("div", { className: "score-cell-label", text: label }),
      el("div", { className: "score-cell-value", text: val }),
    ])
  );
}

export async function renderForYouView(container, _params, onProfileUpdated) {
  clearNode(container);

  const isEventPulse = state.algorithm === "EventPulse";

  // Toolbar controls
  const algoSelect = el(
    "select",
    {
      className: "select-input",
      "aria-label": "Recommendation Algorithm",
      onChange: (e) => {
        setState({ algorithm: e.target.value });
        renderForYouView(container, _params, onProfileUpdated);
      },
    },
    [
      el("option", { value: "EventPulse", text: "EventPulse (Proposed Novelty + MMR)", selected: state.algorithm === "EventPulse" }),
      el("option", { value: "Baseline_A", text: "Baseline A: Chronological", selected: state.algorithm === "Baseline_A" }),
      el("option", { value: "Baseline_B", text: "Baseline B: Content-Based", selected: state.algorithm === "Baseline_B" }),
      el("option", { value: "Baseline_C", text: "Baseline C: Event-Aware (No Novelty)", selected: state.algorithm === "Baseline_C" }),
    ]
  );

  const modeSelect = el(
    "select",
    {
      className: "select-input",
      disabled: !isEventPulse,
      "aria-label": "Adaptive Mode",
      onChange: (e) => {
        const nextMode = e.target.value;
        setState({
          mode: nextMode,
          useCustomWeights: false,
          weights: { ...MODE_PRESET_WEIGHTS[nextMode] },
        });
        renderForYouView(container, _params, onProfileUpdated);
      },
    },
    [
      el("option", { value: "RELEVANCE_FOCUSED", text: "Relevance-Focused (Default)", selected: state.mode === "RELEVANCE_FOCUSED" }),
      el("option", { value: "BALANCED_INTELLIGENCE", text: "Balanced Intelligence", selected: state.mode === "BALANCED_INTELLIGENCE" }),
      el("option", { value: "DISCOVERY_FOCUSED", text: "Discovery-Focused", selected: state.mode === "DISCOVERY_FOCUSED" }),
    ]
  );

  const topKSelect = el(
    "select",
    {
      className: "select-input",
      "aria-label": "Results Count",
      onChange: (e) => {
        setState({ topK: Number(e.target.value) });
        renderForYouView(container, _params, onProfileUpdated);
      },
    },
    [5, 10, 20, 30].map((k) =>
      el("option", { value: String(k), text: `${k} events`, selected: state.topK === k })
    )
  );

  const weightsToggleBtn = el("button", {
    type: "button",
    className: `btn ${state.showWeightsDrawer ? "btn-primary" : ""}`,
    disabled: !isEventPulse,
    text: state.showWeightsDrawer ? "Hide Weights & MMR" : "Weights & MMR",
    onClick: () => {
      setState({ showWeightsDrawer: !state.showWeightsDrawer });
      renderForYouView(container, _params, onProfileUpdated);
    },
  });

  container.appendChild(
    el("div", { className: "toolbar" }, [
      el("div", { className: "toolbar-group" }, [
        el("span", { className: "field-label", text: "Algorithm" }),
        algoSelect,
      ]),
      el("div", { className: "toolbar-group" }, [
        el("span", { className: "field-label", text: "Adaptive Mode" }),
        modeSelect,
      ]),
      el("div", { className: "toolbar-group", style: "min-width: 120px;" }, [
        el("span", { className: "field-label", text: "Feed Size" }),
        topKSelect,
      ]),
      el("div", { className: "toolbar-group", style: "min-width: auto;" }, [
        el("span", { className: "field-label", text: "Tuning" }),
        weightsToggleBtn,
      ]),
    ])
  );

  // Collapsible Weights & MMR Drawer
  if (isEventPulse && state.showWeightsDrawer) {
    const draftWeights = { ...state.weights };
    const sliders = WEIGHT_LABELS.map(([key, label]) => {
      const valReadout = el("span", { className: "mono", text: Number(draftWeights[key]).toFixed(2) });
      const slider = el("input", {
        type: "range",
        min: "0",
        max: "1",
        step: "0.05",
        value: String(draftWeights[key]),
        onInput: (e) => {
          draftWeights[key] = Number(e.target.value);
          valReadout.textContent = draftWeights[key].toFixed(2);
        },
      });
      return el("div", { className: "slider-control" }, [
        el("div", { className: "ingestion-row" }, [
          el("span", { text: label, style: "color: var(--text-secondary);" }),
          valReadout,
        ]),
        slider,
      ]);
    });

    container.appendChild(
      el("div", { className: "weights-drawer" }, [
        el("div", { className: "panel-header" }, [
          el("h2", { className: "panel-title", text: "EventPulse Multi-Objective Scoring & MMR" }),
          el("div", { className: "action-group" }, [
            el("button", {
              type: "button",
              className: "btn btn-sm",
              text: "Reset Mode Defaults",
              onClick: () => {
                setState({
                  useCustomWeights: false,
                  weights: { ...MODE_PRESET_WEIGHTS[state.mode] },
                });
                renderForYouView(container, _params, onProfileUpdated);
              },
            }),
            el("button", {
              type: "button",
              className: "btn btn-primary btn-sm",
              text: "Apply Weights",
              onClick: () => {
                setState({ useCustomWeights: true, weights: { ...draftWeights } });
                renderForYouView(container, _params, onProfileUpdated);
              },
            }),
          ]),
        ]),
        el("div", { className: "weights-grid" }, sliders),
      ])
    );
  }

  const feedHost = el("div");
  feedHost.appendChild(renderCardSkeletons(3));
  container.appendChild(feedHost);

  const reqParams = {
    user_id: state.selectedUserId,
    algorithm: state.algorithm,
    top_k: state.topK,
  };
  if (isEventPulse) {
    reqParams.mode = state.mode;
    if (state.useCustomWeights) {
      Object.assign(reqParams, state.weights);
    }
  }

  try {
    const data = await api.getRecommendations(reqParams);
    clearNode(feedHost);

    if (data.is_cold_start) {
      feedHost.appendChild(
        el("div", {
          className: "notice-banner",
          text: `Persona '${data.user_id}' is currently in cold-start (< 3 interactions). Use Like or Read below to build real-time interest centroids.`,
        })
      );
    }

    if (!data.recommendations || data.recommendations.length === 0) {
      feedHost.appendChild(
        renderEmptyState(
          "No recommendations available",
          "No candidate events matched the active database partition."
        )
      );
      return;
    }

    data.recommendations.forEach((rec) => {
      const sendInteraction = async (type, btn) => {
        btn.disabled = true;
        try {
          const resp = await api.postInteraction({
            user_id: state.selectedUserId,
            article_id: rec.representative_article_id,
            event_id: rec.event_id,
            interaction_type: type,
          });
          showToast(
            `Recorded ${type} (${resp.updated_interaction_count} total interactions for ${resp.user_id}).`,
            "success"
          );
          if (onProfileUpdated) onProfileUpdated();
          renderForYouView(container, _params, onProfileUpdated);
        } catch (err) {
          showToast(err.message || "Failed to record interaction", "error");
          btn.disabled = false;
        }
      };

      const likeBtn = el("button", { type: "button", className: "btn btn-sm", text: "Like" });
      likeBtn.addEventListener("click", () => sendInteraction("LIKE", likeBtn));

      const readBtn = el("button", { type: "button", className: "btn btn-sm btn-primary", text: "Mark Read" });
      readBtn.addEventListener("click", () => sendInteraction("READ", readBtn));

      const dislikeBtn = el("button", { type: "button", className: "btn btn-sm", text: "Dislike" });
      dislikeBtn.addEventListener("click", () => sendInteraction("DISLIKE", dislikeBtn));

      const sourceLinkContainer = el("span", { className: "action-group" }, [
        el("button", {
          type: "button",
          className: "btn btn-sm",
          text: "Inspect Timeline",
          onClick: () => navigateTo("explore", { event: rec.event_id }),
        }),
        el("button", {
          type: "button",
          className: "btn btn-sm",
          text: "Original Link",
          onClick: async (e) => {
            const btn = e.currentTarget;
            btn.disabled = true;
            try {
              const arts = await api.getEventArticles(rec.event_id);
              const match =
                arts.find((a) => a.id === rec.representative_article_id) || arts[arts.length - 1];
              const safeHref = safeExternalUrl(match?.url);
              if (safeHref) {
                window.open(safeHref, "_blank", "noopener,noreferrer");
              } else {
                showToast("No valid external HTTP(S) URL recorded for this article.", "info");
              }
            } catch (err) {
              showToast(err.message || "Could not resolve article URL.", "error");
            } finally {
              btn.disabled = false;
            }
          },
        }),
      ]);

      const card = el("article", { className: "story-card" }, [
        el("div", { className: "story-meta-row" }, [
          el("div", {}, [
            el("span", { className: "story-rank", text: `#${rec.rank}` }),
            " · ",
            el("strong", { text: rec.source_name || "Unknown Source", style: "color: var(--text-secondary);" }),
            " · ",
            el("span", { className: "mono", text: formatUtcTimestamp(rec.published_at) }),
          ]),
          el("span", { className: "badge badge-subtle mono", text: rec.event_id }),
        ]),
        el("h3", { className: "story-headline", text: rec.article_headline }),
        el("p", {
          className: "story-summary",
          text: rec.summary || "Multi-source event cluster without article abstract.",
        }),
        el("div", { className: "explanation-note" }, [
          el("strong", { text: "Grounded Explanation: " }),
          rec.explanation,
        ]),
        el("div", { className: "score-strip" }, makeScoreCells(rec, data.algorithm)),
        el("div", { className: "story-actions" }, [
          el("div", { className: "action-group" }, [likeBtn, readBtn, dislikeBtn]),
          sourceLinkContainer,
        ]),
      ]);

      feedHost.appendChild(card);
    });
  } catch (err) {
    clearNode(feedHost);
    feedHost.appendChild(
      renderErrorState(err.message || "Failed to fetch recommendations.", () =>
        renderForYouView(container, _params, onProfileUpdated)
      )
    );
  }
}
