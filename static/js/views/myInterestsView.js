/** View E: My Interests — Synthetic Persona Profile Stats & Interaction Log. */

import { api } from "../api.js";
import { navigateTo } from "../router.js";
import { state } from "../state.js";
import { renderCardSkeletons, renderEmptyState, renderErrorState } from "../components/skeletons.js";
import { clearNode, el, formatUtcTimestamp } from "../utils/dom.js";

export async function renderMyInterestsView(container) {
  clearNode(container);

  container.appendChild(
    el("div", { className: "notice-banner" }, [
      el("strong", { text: "Synthetic Demonstration Persona Notice: " }),
      "Preset accounts (user_tech, user_geopolitics, user_science, user_cold_start) are live demonstration personas for testing dynamic profile adaptation in SQLite. They are strictly separate from offline MIND research benchmark evaluations.",
    ])
  );

  const host = el("div");
  host.appendChild(renderCardSkeletons(2));
  container.appendChild(host);

  try {
    const [profile, interactions] = await Promise.all([
      api.getUserProfile(state.selectedUserId),
      api.getUserInteractions(state.selectedUserId, 50),
    ]);

    clearNode(host);

    const activePreset = state.presets.find((p) => p.user_id === profile.user_id);

    host.appendChild(
      el("div", { className: "panel", style: "margin-bottom: 18px;" }, [
        el("div", { className: "panel-header" }, [
          el("div", {}, [
            el("h3", {
              className: "panel-title",
              text: profile.display_name || profile.user_id,
            }),
            activePreset
              ? el("div", {
                  text: activePreset.description,
                  style: "font-size: 12.5px; color: var(--text-muted); margin-top: 3px;",
                })
              : null,
          ]),
          el("span", {
            className: profile.is_synthetic_demo ? "badge badge-accent" : "badge badge-subtle",
            text: profile.is_synthetic_demo ? "Synthetic Demo Persona" : "Custom User",
          }),
        ]),
        el("div", { className: "score-strip" }, [
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "User ID" }),
            el("div", { className: "score-cell-value mono", text: profile.user_id }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Total Interactions" }),
            el("div", { className: "score-cell-value mono", text: String(profile.interaction_count) }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Consumed Sentences" }),
            el("div", { className: "score-cell-value mono", text: String(profile.consumed_sentences_count) }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Profile State" }),
            el("div", {
              className: "score-cell-value",
              text: profile.is_cold_start ? "COLD START (< 3)" : "ACTIVE DUAL CENTROID",
              style: profile.is_cold_start ? "color: var(--status-warn-text);" : "color: var(--status-new-text);",
            }),
          ]),
        ]),
      ])
    );

    const histPanel = el("div", { className: "panel" }, [
      el("div", { className: "panel-header" }, [
        el("h3", { className: "panel-title", text: `Recent Engagement History (${interactions.length})` }),
        el("button", {
          type: "button",
          className: "btn btn-sm btn-primary",
          text: "Go to For You Feed",
          onClick: () => navigateTo("for-you"),
        }),
      ]),
    ]);

    if (!interactions.length) {
      histPanel.appendChild(
        renderEmptyState(
          "No Recorded Interactions Yet",
          "Interact with articles on the For You page (Like, Mark Read, Dislike) to build this persona's dual long-term and short-term semantic profile."
        )
      );
    } else {
      const tbody = el("tbody");
      for (const r of interactions) {
        let actionCls = "badge badge-subtle";
        if (r.interaction_type === "READ" || r.interaction_type === "LIKE") actionCls = "badge badge-new";
        else if (r.interaction_type === "DISLIKE") actionCls = "badge badge-repeated";

        tbody.appendChild(
          el("tr", {}, [
            el("td", {}, [el("span", { className: actionCls, text: r.interaction_type })]),
            el("td", { className: "mono", text: Number(r.interaction_weight).toFixed(1) }),
            el("td", { className: "mono", text: formatUtcTimestamp(r.interaction_time) }),
            el("td", { className: "mono", text: r.article_id ?? "—" }),
            el("td", {}, [
              r.event_id
                ? el("button", {
                    type: "button",
                    className: "btn btn-sm",
                    text: r.event_id,
                    onClick: () => navigateTo("explore", { event: r.event_id }),
                  })
                : "—",
            ]),
          ])
        );
      }

      histPanel.appendChild(
        el("div", { className: "table-wrap" }, [
          el("table", { className: "data-table" }, [
            el("thead", {}, [
              el("tr", {}, [
                el("th", { text: "Action" }),
                el("th", { text: "Weight" }),
                el("th", { text: "Timestamp" }),
                el("th", { text: "Article ID" }),
                el("th", { text: "Event Cluster" }),
              ]),
            ]),
            tbody,
          ]),
        ])
      );
    }

    host.appendChild(histPanel);
  } catch (err) {
    clearNode(host);
    host.appendChild(
      renderErrorState(err.message || "Failed to load user profile.", () => renderMyInterestsView(container))
    );
  }
}
