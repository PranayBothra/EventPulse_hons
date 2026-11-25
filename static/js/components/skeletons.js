/** Loading skeletons, empty state, and error state components. */

import { el } from "../utils/dom.js";

export function renderCardSkeletons(count = 3) {
  const wrap = el("div");
  for (let i = 0; i < count; i += 1) {
    wrap.appendChild(el("div", { className: "skeleton skeleton-card" }));
  }
  return wrap;
}

export function renderEmptyState(title, message, actionBtn = null) {
  return el("div", { className: "state-box" }, [
    el("strong", { text: title, style: "display:block; margin-bottom:6px; color: var(--text-primary);" }),
    el("div", { text: message, style: "font-size: 13.5px;" }),
    actionBtn ? el("div", { style: "margin-top: 12px;" }, [actionBtn]) : null,
  ]);
}

export function renderErrorState(message, onRetry = null) {
  return el("div", { className: "state-box error" }, [
    el("strong", { text: "Unable to load data", style: "display:block; margin-bottom:6px;" }),
    el("div", { text: message, style: "font-size: 13.5px;" }),
    onRetry
      ? el("div", { style: "margin-top: 12px;" }, [
          el("button", { className: "btn btn-sm", type: "button", text: "Retry", onClick: onRetry }),
        ])
      : null,
  ]);
}
