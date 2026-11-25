/** View C: Novelty Inspector — Sentence-Level Evidence vs Chronological Event Memory. */

import { api } from "../api.js";
import { setState, state } from "../state.js";
import { renderCardSkeletons, renderEmptyState, renderErrorState } from "../components/skeletons.js";
import { clearNode, el, evidenceBadgeClass, formatUtcTimestamp } from "../utils/dom.js";

async function runInspection(resultHost, eventId, articleId = null) {
  clearNode(resultHost);
  if (!eventId) {
    resultHost.appendChild(renderEmptyState("No Evolving Event Selected", "Select an event with at least 2 articles."));
    return;
  }

  resultHost.appendChild(renderCardSkeletons(2));
  try {
    const params = { event_id: eventId, persist: true };
    if (articleId) params.article_id = articleId;

    const data = await api.inspectNovelty(params);
    clearNode(resultHost);

    resultHost.appendChild(
      el("div", { className: "panel", style: "margin-bottom: 16px;" }, [
        el("div", { className: "panel-header" }, [
          el("div", {}, [
            el("h3", { className: "panel-title", text: data.article_headline }),
            el("div", {
              className: "mono",
              text: `${data.source_name} · ${formatUtcTimestamp(data.published_at)} · Event ${data.event_id}`,
              style: "font-size: 12px; color: var(--text-faint); margin-top: 3px;",
            }),
          ]),
        ]),
        el("div", { className: "score-strip" }, [
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Overall Event Novelty" }),
            el("div", { className: "score-cell-value", text: Number(data.event_novelty_score).toFixed(3) }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Novel Sentences" }),
            el("div", { className: "score-cell-value", text: `${data.novel_sentence_count} / ${data.total_sentence_count}` }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Repeated Sentences" }),
            el("div", { className: "score-cell-value", text: `${data.redundant_sentence_count} / ${data.total_sentence_count}` }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Insufficient Evidence" }),
            el("div", { className: "score-cell-value", text: `${data.insufficient_evidence_count} / ${data.total_sentence_count}` }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Prior Memory H(e,t)" }),
            el("div", { className: "score-cell-value", text: `${data.prior_memory_sentence_count} sents` }),
          ]),
        ]),
      ])
    );

    if (!data.sentence_results || !data.sentence_results.length) {
      resultHost.appendChild(renderEmptyState("No Extracted Sentences", "No information-bearing sentences were extracted for this article."));
      return;
    }

    const listHost = el("div", { className: "panel" }, [
      el("div", { className: "panel-header" }, [
        el("h3", { className: "panel-title", text: "Sentence-Level Evidence Breakdown" }),
      ]),
    ]);

    for (const s of data.sentence_results) {
      listHost.appendChild(
        el("div", { className: "sentence-card" }, [
          el("div", { className: "story-meta-row" }, [
            el("span", { className: evidenceBadgeClass(s.evidence_status), text: s.evidence_status }),
            el("span", {
              className: "mono",
              text: `Sentence #${s.sentence_index} · Info ${Number(s.informativeness_score).toFixed(2)} · Max Sim ${Number(s.highest_similarity).toFixed(3)} · Novelty ${Number(s.novelty_score).toFixed(2)}`,
            }),
          ]),
          el("div", { className: "sentence-text", text: s.sentence_text }),
          s.most_similar_text
            ? el("div", { className: "sentence-prior" }, [
                el("strong", { text: "Closest Prior Sentence in Memory: " }),
                s.most_similar_text,
              ])
            : null,
        ])
      );
    }

    resultHost.appendChild(listHost);
  } catch (err) {
    clearNode(resultHost);
    resultHost.appendChild(
      renderErrorState(err.message || "Novelty inspection failed.", () => runInspection(resultHost, eventId, articleId))
    );
  }
}

export async function renderNoveltyInspectorView(container, params) {
  clearNode(container);

  const reqEvent = params?.get("event");
  if (reqEvent) {
    setState({ noveltyEventId: reqEvent, noveltyArticleId: null });
  }

  const eventSelect = el("select", { className: "select-input", "aria-label": "Evolving Event" });
  const articleSelect = el("select", { className: "select-input", "aria-label": "Target Article" });

  container.appendChild(
    el("div", { className: "toolbar" }, [
      el("div", { className: "toolbar-group grow" }, [
        el("span", { className: "field-label", text: "Evolving Multi-Article Event" }),
        eventSelect,
      ]),
      el("div", { className: "toolbar-group grow" }, [
        el("span", { className: "field-label", text: "Article to Inspect Against Earlier Memory" }),
        articleSelect,
      ]),
    ])
  );

  const resultHost = el("div");
  resultHost.appendChild(renderCardSkeletons(2));
  container.appendChild(resultHost);

  try {
    const evolving = await api.getEvolvingEvents({ min_articles: 2, limit: 50 });
    if (!evolving.length) {
      clearNode(resultHost);
      resultHost.appendChild(
        renderEmptyState("No Multi-Article Events", "Cluster at least two related articles into an event to inspect incremental novelty.")
      );
      return;
    }

    let chosenEvent = evolving.find((e) => e.id === state.noveltyEventId)?.id || evolving[0].id;
    setState({ noveltyEventId: chosenEvent });

    clearNode(eventSelect);
    for (const ev of evolving) {
      eventSelect.appendChild(
        el("option", {
          value: ev.id,
          text: `${ev.title} (${ev.article_count} articles)`,
          selected: ev.id === chosenEvent,
        })
      );
    }

    const loadArticlesAndInspect = async (evId, explicitArtId = null) => {
      const arts = await api.getEventArticles(evId);
      clearNode(articleSelect);
      if (!arts.length) {
        await runInspection(resultHost, evId, null);
        return;
      }
      const chosenArtId = explicitArtId || arts[arts.length - 1].id;
      setState({ noveltyArticleId: chosenArtId });

      arts.forEach((a, idx) => {
        articleSelect.appendChild(
          el("option", {
            value: String(a.id),
            text: `#${idx + 1} [${formatUtcTimestamp(a.published_at)}] ${a.source_name}: ${a.headline}`,
            selected: a.id === chosenArtId,
          })
        );
      });

      await runInspection(resultHost, evId, chosenArtId);
    };

    eventSelect.addEventListener("change", (e) => {
      chosenEvent = e.target.value;
      setState({ noveltyEventId: chosenEvent, noveltyArticleId: null });
      loadArticlesAndInspect(chosenEvent, null);
    });

    articleSelect.addEventListener("change", (e) => {
      const aid = Number(e.target.value);
      setState({ noveltyArticleId: aid });
      runInspection(resultHost, chosenEvent, aid);
    });

    await loadArticlesAndInspect(chosenEvent, state.noveltyArticleId);
  } catch (err) {
    clearNode(resultHost);
    resultHost.appendChild(
      renderErrorState(err.message || "Failed to load evolving events.", () =>
        renderNoveltyInspectorView(container, params)
      )
    );
  }
}
