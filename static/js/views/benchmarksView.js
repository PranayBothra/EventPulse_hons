/** View F: Benchmarks — Offline MIND Ranking & Corpus Novelty Saturation Reports. */

import { api } from "../api.js";
import { renderGroupedBarChart } from "../components/charts.js";
import { renderCardSkeletons, renderErrorState } from "../components/skeletons.js";
import { clearNode, el, formatUtcTimestamp } from "../utils/dom.js";

function fmtMetricCell(metricObj) {
  if (metricObj === null || metricObj === undefined) return "—";
  if (typeof metricObj === "number") return metricObj.toFixed(4);
  if (typeof metricObj === "object" && metricObj.mean !== undefined) {
    const ci = Array.isArray(metricObj.ci_95)
      ? `[${Number(metricObj.ci_95[0]).toFixed(3)}, ${Number(metricObj.ci_95[1]).toFixed(3)}]`
      : "";
    return el("div", {}, [
      el("div", { className: "mono", text: Number(metricObj.mean).toFixed(4), style: "font-weight: 600; color: var(--text-primary);" }),
      ci ? el("div", { className: "mono", text: ci, style: "font-size: 10.5px; color: var(--text-faint);" }) : null,
    ]);
  }
  return String(metricObj);
}

export async function renderBenchmarksView(container) {
  clearNode(container);

  container.appendChild(
    el("div", { className: "notice-banner" }, [
      el("strong", { text: "Offline Research Benchmarks: " }),
      "Metrics below are loaded directly from archived evaluation artifacts (reports/mind_benchmark_results.json and reports/corpus_novelty_results.json). They reflect offline experimental splits, not live production click telemetry.",
    ])
  );

  const host = el("div");
  host.appendChild(renderCardSkeletons(2));
  container.appendChild(host);

  try {
    const [mindPayload, corpusPayload] = await Promise.all([
      api.getMindBenchmark(),
      api.getCorpusNoveltyBenchmark(),
    ]);

    clearNode(host);

    const mindRes = mindPayload.results || {};
    const primaryModels = mindRes.primary_models || mindRes;
    const metricKeys = ["NDCG@5", "NDCG@10", "MRR", "AUC", "ILD@5", "Redundancy@5"];

    // 1. MIND Comparison Table
    const tbody = el("tbody");
    const chartModels = [];
    for (const [modelName, metrics] of Object.entries(primaryModels)) {
      if (!metrics || typeof metrics !== "object" || !metrics["NDCG@5"]) continue;
      tbody.appendChild(
        el("tr", {}, [
          el("td", { text: modelName, style: "font-weight: 600; color: var(--text-primary);" }),
          ...metricKeys.map((k) => el("td", {}, [fmtMetricCell(metrics[k])])),
        ])
      );
      chartModels.push({
        label: modelName,
        values: Object.fromEntries(
          ["NDCG@5", "MRR", "AUC", "ILD@5", "Redundancy@5"].map((k) => [
            k,
            typeof metrics[k] === "object" ? metrics[k].mean : metrics[k],
          ])
        ),
      });
    }

    host.appendChild(
      el("div", { className: "panel", style: "margin-bottom: 18px;" }, [
        el("div", { className: "panel-header" }, [
          el("div", {}, [
            el("h3", { className: "panel-title", text: "1. MIND Recommendation Evaluation (Offline Split)" }),
            el("div", {
              className: "mono",
              text: `Scope: ${mindPayload.evaluation_scope} · Split: ${mindRes.split || "dev"} · Sessions: ${mindRes.eval_sessions_analyzed || "—"} · Encoder: ${mindRes.embedding_model || "—"} · Audited: ${formatUtcTimestamp(mindRes.audit_timestamp)}`,
              style: "font-size: 11.5px; color: var(--text-faint); margin-top: 3px;",
            }),
          ]),
          el("span", { className: "badge badge-subtle", text: mindPayload.source_file }),
        ]),
        el("div", { className: "table-wrap" }, [
          el("table", { className: "data-table" }, [
            el("thead", {}, [
              el("tr", {}, [
                el("th", { text: "Model" }),
                ...metricKeys.map((k) => el("th", { text: k })),
              ]),
            ]),
            tbody,
          ]),
        ]),
      ])
    );

    // 2. SVG Comparison Chart
    if (chartModels.length) {
      host.appendChild(
        el("div", { className: "panel", style: "margin-bottom: 18px;" }, [
          el("div", { className: "panel-header" }, [
            el("h3", { className: "panel-title", text: "2. Model Metric Profile Comparison (NDCG@5, MRR, AUC, ILD@5, Redundancy@5)" }),
          ]),
          el("div", { className: "chart-box" }, [
            renderGroupedBarChart(chartModels, ["NDCG@5", "MRR", "AUC", "ILD@5", "Redundancy@5"]),
          ]),
        ])
      );
    }

    // 3. Corpus Novelty Saturation Panel
    const cRes = corpusPayload.results || {};
    const meanInit = Number(cRes.mean_initial_novelty ?? 0);
    const meanFollow = Number(cRes.mean_followup_novelty ?? 0);
    const pctDelta = meanInit > 0 ? (((meanFollow - meanInit) / meanInit) * 100).toFixed(2) : "—";
    const evCounts = cRes.evidence_counts || {};
    const evProps = cRes.evidence_proportions || {};

    host.appendChild(
      el("div", { className: "panel" }, [
        el("div", { className: "panel-header" }, [
          el("div", {}, [
            el("h3", { className: "panel-title", text: "3. Event Corpus Novelty Saturation Analysis (Offline)" }),
            el("div", {
              className: "mono",
              text: `Source: ${corpusPayload.source_file} · Evaluation Scope: ${corpusPayload.evaluation_scope}`,
              style: "font-size: 11.5px; color: var(--text-faint); margin-top: 3px;",
            }),
          ]),
        ]),
        el("div", { className: "score-strip" }, [
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Events Evaluated" }),
            el("div", { className: "score-cell-value mono", text: String(cRes.total_events_evaluated ?? "—") }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Sentences Classified" }),
            el("div", { className: "score-cell-value mono", text: String(cRes.total_sentences_classified ?? "—") }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Initial Mean Novelty" }),
            el("div", { className: "score-cell-value mono", text: meanInit.toFixed(4) }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Follow-Up Mean Novelty" }),
            el("div", { className: "score-cell-value mono", text: meanFollow.toFixed(4) }),
          ]),
          el("div", { className: "score-cell" }, [
            el("div", { className: "score-cell-label", text: "Relative Change" }),
            el("div", { className: "score-cell-value mono", text: `${pctDelta}%` }),
          ]),
        ]),
        el(
          "div",
          { className: "action-group", style: "margin-top: 10px;" },
          ["POTENTIALLY_NEW", "REPEATED_INFORMATION", "INSUFFICIENT_EVIDENCE"].map((k) =>
            el("span", {
              className: "badge badge-subtle mono",
              text: `${k}: ${evCounts[k] ?? 0} (${((evProps[k] ?? 0) * 100).toFixed(1)}%)`,
            })
          )
        ),
      ])
    );
  } catch (err) {
    clearNode(host);
    host.appendChild(
      renderErrorState(err.message || "Failed to load offline benchmarks.", () => renderBenchmarksView(container))
    );
  }
}
