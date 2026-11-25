/** Non-blocking toast notification helper. */

import { el } from "../utils/dom.js";

const region = () => document.getElementById("toast-region");

export function showToast(message, type = "info", durationMs = 4200) {
  const host = region();
  if (!host) return;
  const cls = type === "error" ? "toast error" : type === "success" ? "toast success" : "toast";
  const item = el("div", { className: cls, text: message });
  host.appendChild(item);
  setTimeout(() => {
    if (item.parentNode) {
      item.parentNode.removeChild(item);
    }
  }, durationMs);
}
