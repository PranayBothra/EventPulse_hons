/** Safe DOM helpers preventing raw HTML injection from dynamic news strings. */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs || {})) {
    if (val === undefined || val === null) continue;
    if (key === "className") {
      node.className = val;
    } else if (key === "text") {
      node.textContent = String(val);
    } else if (key.startsWith("on") && typeof val === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), val);
    } else if (key === "dataset" && typeof val === "object") {
      for (const [dk, dv] of Object.entries(val)) {
        if (dv !== undefined && dv !== null) {
          node.dataset[dk] = String(dv);
        }
      }
    } else if (typeof val === "boolean") {
      if (val) node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(val));
    }
  }
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    if (typeof child === "string" || typeof child === "number") {
      node.appendChild(document.createTextNode(String(child)));
    } else {
      node.appendChild(child);
    }
  }
  return node;
}

export function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

export function safeExternalUrl(rawUrl) {
  if (!rawUrl || typeof rawUrl !== "string") return null;
  const trimmed = rawUrl.trim();
  if (!trimmed) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol === "https:" || parsed.protocol === "http:") {
      return parsed.href;
    }
    return null;
  } catch {
    return null;
  }
}

export function formatUtcTimestamp(isoStr) {
  if (!isoStr || typeof isoStr !== "string") return "Unknown time";
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) {
    return isoStr.slice(0, 16).replace("T", " ");
  }
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
}

export function evidenceBadgeClass(status) {
  if (status === "POTENTIALLY_NEW") return "badge badge-new";
  if (status === "REPEATED_INFORMATION") return "badge badge-repeated";
  return "badge badge-insufficient";
}
