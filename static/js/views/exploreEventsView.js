/** View B: Explore Events — Cluster Table & Chronological Coverage Timeline. */

import { api } from "../api.js";
import { navigateTo } from "../router.js";
import { setState, state } from "../state.js";
import { renderCardSkeletons, renderEmptyState, renderErrorState } from "../components/skeletons.js";
import { clearNode, el, formatUtcTimestamp, safeExternalUrl } from "../utils/dom.js";

function statusBadge(status) {
  const s = (status || "ACTIVE").toUpperCase();
  let cls = "badge badge-subtle";
  if (s === "EMERGING") cls = "badge badge-accent";
  else if (s === "ACTIVE") cls = "badge badge-new";
  else if (s === "DORMANT") cls = "badge badge-warn";
  return el("span", { className: cls, text: s });
}

async function renderTimeline(timelineHost, eventSummary) {
  clearNode(timelineHost);
  if (!eventSummary) {
    timelineHost.appendChild(
      renderEmptyState("Select an Event", "Choose any event row to inspect its chronological multi-source article timeline.")
    );
    return;
  }

  timelineHost.appendChild(renderCardSkeletons(2));
  try {
    const articles = await api.getEventArticles(eventSummary.id);
    clearNode(timelineHost);

    const headerActions = [
      el("span", { className: "badge badge-subtle mono", text: `${articles.length} articles` }),
    ];
    if (articles.length >= 2) {
      headerActions.push(
        el("button", {
          type: "button",
          className: "btn btn-primary btn-sm",
          text: "Inspect Novelty",
          onClick: () => navigateTo("novelty", { event: eventSummary.id }),
        })
      );
    }

    const panel = el("div", { className: "panel" }, [
      el("div", { className: "panel-header" }, [
        el("div", {}, [
          el("h3", { className: "panel-title", text: eventSummary.title }),
          el("div", {
            className: "mono",
            text: `${eventSummary.id} · ${formatUtcTimestamp(eventSummary.first_seen_at)} -> ${formatUtcTimestamp(eventSummary.last_seen_at)}`,
            style: "font-size: 11.5px; color: var(--text-faint); margin-top: 3px;",
          }),
        ]),
        el("div", { className: "action-group" }, headerActions),
      ]),
    ]);

    if (!articles.length) {
      panel.appendChild(renderEmptyState("No Articles Linked", "This event cluster has no linked article rows."));
      timelineHost.appendChild(panel);
      return;
    }

    const list = el("div", { className: "timeline" });
    for (const art of articles) {
      const safeHref = safeExternalUrl(art.url);
      const isLiveFn = art.provider === "freenewsapi";
      const providerBadge = el("span", {
        className: isLiveFn ? "badge badge-accent" : "badge badge-subtle",
        text: isLiveFn ? "Live · FreeNewsAPI" : "RSS / Corpus",
      });

      const novText =
        art.event_novelty_score !== null && art.event_novelty_score !== undefined
          ? `Event Novelty: ${Number(art.event_novelty_score).toFixed(3)}`
          : "Event Novelty: Not yet evaluated";

      const metaBottom = [
        el("span", { className: "mono", text: novText }),
        art.similarity_score !== null && art.similarity_score !== undefined
          ? el("span", { className: "mono", text: `Cluster Sim: ${Number(art.similarity_score).toFixed(3)}` })
          : null,
        safeHref
          ? el("a", {
              href: safeHref,
              target: "_blank",
              rel: "noopener noreferrer",
              className: "external-link",
              text: "Open Source Article ↗",
            })
          : null,
      ];

      list.appendChild(
        el("div", { className: "timeline-item" }, [
          el("div", { className: "story-meta-row" }, [
            el("div", {}, [
              el("strong", { text: art.source_name || art.source_id, style: "color: var(--text-primary);" }),
              " · ",
              el("span", { className: "mono", text: formatUtcTimestamp(art.published_at) }),
            ]),
            providerBadge,
          ]),
          el("div", {
            text: art.headline,
            style: "font-family: var(--font-serif); font-size: 16px; font-weight: 600; color: var(--text-primary); margin-bottom: 5px;",
          }),
          art.description
            ? el("div", { text: art.description, style: "font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;" })
            : null,
          el("div", { className: "action-group", style: "font-size: 11.5px; color: var(--text-faint);" }, metaBottom),
        ])
      );
    }

    panel.appendChild(list);
    timelineHost.appendChild(panel);
  } catch (err) {
    clearNode(timelineHost);
    timelineHost.appendChild(
      renderErrorState(err.message || "Failed to load article timeline.", () => renderTimeline(timelineHost, eventSummary))
    );
  }
}

export async function renderExploreEventsView(container, params) {
  clearNode(container);

  const requestedEventId = params?.get("event");
  if (requestedEventId) {
    setState({ selectedEventId: requestedEventId });
  }

  const searchInput = el("input", {
    type: "search",
    className: "text-input",
    placeholder: "Search loaded events by title or ID...",
    value: state.eventSearch,
  });

  const statusSelect = el(
    "select",
    {
      className: "select-input",
      onChange: (e) => {
        setState({ eventStatusFilter: e.target.value });
        renderExploreEventsView(container, new URLSearchParams());
      },
    },
    [
      el("option", { value: "", text: "All Statuses", selected: state.eventStatusFilter === "" }),
      el("option", { value: "ACTIVE", text: "ACTIVE", selected: state.eventStatusFilter === "ACTIVE" }),
      el("option", { value: "EMERGING", text: "EMERGING", selected: state.eventStatusFilter === "EMERGING" }),
      el("option", { value: "DORMANT", text: "DORMANT", selected: state.eventStatusFilter === "DORMANT" }),
      el("option", { value: "INACTIVE", text: "INACTIVE", selected: state.eventStatusFilter === "INACTIVE" }),
    ]
  );

  const sortSelect = el(
    "select",
    {
      className: "select-input",
      onChange: (e) => {
        setState({ eventSortBy: e.target.value });
        renderExploreEventsView(container, new URLSearchParams());
      },
    },
    [
      el("option", { value: "article_count", text: "Largest Clusters", selected: state.eventSortBy === "article_count" }),
      el("option", { value: "last_seen_at", text: "Most Recent Activity", selected: state.eventSortBy === "last_seen_at" }),
    ]
  );

  container.appendChild(
    el("div", { className: "toolbar" }, [
      el("div", { className: "toolbar-group grow" }, [
        el("span", { className: "field-label", text: "Filter Loaded Clusters" }),
        searchInput,
      ]),
      el("div", { className: "toolbar-group" }, [
        el("span", { className: "field-label", text: "Lifecycle Status" }),
        statusSelect,
      ]),
      el("div", { className: "toolbar-group" }, [
        el("span", { className: "field-label", text: "Sort Order" }),
        sortSelect,
      ]),
    ])
  );

  const split = el("div", { className: "two-col-split" });
  const tableHost = el("div");
  const timelineHost = el("div");
  tableHost.appendChild(renderCardSkeletons(2));
  split.appendChild(tableHost);
  split.appendChild(timelineHost);
  container.appendChild(split);

  try {
    let events = await api.getEvents({
      limit: 100,
      status: state.eventStatusFilter || undefined,
      sort_by: state.eventSortBy,
    });

    // If URL specified a specific event ID outside the top 100 slice, fetch its summary so timeline still opens
    let activeEv = events.find((ev) => ev.id === state.selectedEventId) || events[0] || null;
    if (requestedEventId && !events.some((ev) => ev.id === requestedEventId)) {
      activeEv = {
        id: requestedEventId,
        title: requestedEventId,
        first_seen_at: "",
        last_seen_at: "",
      };
    }

    const drawTable = () => {
      clearNode(tableHost);
      const query = (state.eventSearch || "").trim().toLowerCase();
      const filtered = query
        ? events.filter(
            (ev) =>
              (ev.title || "").toLowerCase().includes(query) ||
              (ev.id || "").toLowerCase().includes(query)
          )
        : events;

      if (!filtered.length) {
        tableHost.appendChild(renderEmptyState("No Matching Events", "Try clearing your search or lifecycle filter."));
        return;
      }

      const tbody = el("tbody");
      filtered.forEach((ev) => {
        const isSel = activeEv && activeEv.id === ev.id;
        const tr = el(
          "tr",
          {
            className: `clickable ${isSel ? "selected" : ""}`,
            onClick: () => {
              activeEv = ev;
              setState({ selectedEventId: ev.id });
              drawTable();
              renderTimeline(timelineHost, ev);
            },
          },
          [
            el("td", {}, [
              el("div", { text: ev.title, style: "font-weight: 500; color: var(--text-primary);" }),
              el("div", { className: "mono", text: ev.id, style: "font-size: 11px; color: var(--text-faint);" }),
            ]),
            el("td", { className: "mono", text: ev.article_count }),
            el("td", { className: "mono", text: ev.publisher_count }),
            el("td", {}, [statusBadge(ev.status)]),
            el("td", { className: "mono", text: formatUtcTimestamp(ev.last_seen_at), style: "font-size: 11.5px;" }),
          ]
        );
        tbody.appendChild(tr);
      });

      const table = el("table", { className: "data-table" }, [
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "Event Cluster" }),
            el("th", { text: "Arts" }),
            el("th", { text: "Sources" }),
            el("th", { text: "Status" }),
            el("th", { text: "Last Seen" }),
          ]),
        ]),
        tbody,
      ]);

      tableHost.appendChild(
        el("div", { className: "panel" }, [
          el("div", { className: "panel-header" }, [
            el("h3", { className: "panel-title", text: `Event Clusters (${filtered.length})` }),
          ]),
          el("div", { className: "table-wrap" }, [table]),
        ])
      );
    };

    searchInput.addEventListener("input", (e) => {
      setState({ eventSearch: e.target.value });
      drawTable();
    });

    drawTable();
    await renderTimeline(timelineHost, activeEv);
  } catch (err) {
    clearNode(tableHost);
    clearNode(timelineHost);
    tableHost.appendChild(
      renderErrorState(err.message || "Failed to load events.", () => renderExploreEventsView(container, params))
    );
  }
}
