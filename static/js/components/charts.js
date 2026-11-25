/** Clean SVG bar-chart renderer for offline research benchmarks (no external chart library). */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, textContent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    node.setAttribute(k, String(v));
  }
  if (textContent !== null) {
    node.textContent = String(textContent);
  }
  return node;
}

/**
 * Renders a grouped bar chart comparing benchmark models across key IR metrics.
 * models: [{ label, values: { "NDCG@5": 0.2133, ... } }]
 */
export function renderGroupedBarChart(models, metricKeys, height = 250) {
  const width = 680;
  const pad = { top: 24, right: 20, bottom: 52, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const colors = ["#38bdf8", "#3b82f6", "#34d399", "#fbbf24"];
  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height: String(height),
    role: "img",
    "aria-label": "Offline MIND Benchmark Metric Comparison Chart",
  });

  // Grid lines (0.0 to 1.0)
  for (let tick = 0; tick <= 5; tick += 1) {
    const val = tick * 0.2;
    const y = pad.top + plotH * (1 - val);
    svg.appendChild(
      svgEl("line", {
        x1: pad.left,
        x2: width - pad.right,
        y1: y,
        y2: y,
        stroke: "#1e293b",
        "stroke-width": 1,
      })
    );
    svg.appendChild(
      svgEl(
        "text",
        {
          x: pad.left - 8,
          y: y + 4,
          "text-anchor": "end",
          fill: "#64748b",
          "font-size": "11",
          "font-family": "var(--font-mono)",
        },
        val.toFixed(1)
      )
    );
  }

  const groupW = plotW / Math.max(1, metricKeys.length);
  const barW = Math.min(26, (groupW - 18) / Math.max(1, models.length));

  metricKeys.forEach((mKey, gIdx) => {
    const groupCenterX = pad.left + gIdx * groupW + groupW / 2;
    const totalBarsW = models.length * barW + (models.length - 1) * 4;
    const startX = groupCenterX - totalBarsW / 2;

    models.forEach((m, mIdx) => {
      const val = Math.max(0, Math.min(1, Number(m.values[mKey] || 0)));
      const barH = val * plotH;
      const x = startX + mIdx * (barW + 4);
      const y = pad.top + plotH - barH;

      const rect = svgEl("rect", {
        x: x.toFixed(1),
        y: y.toFixed(1),
        width: barW.toFixed(1),
        height: Math.max(1, barH).toFixed(1),
        rx: 3,
        fill: colors[mIdx % colors.length],
        opacity: 0.9,
      });
      rect.appendChild(svgEl("title", {}, `${m.label} — ${mKey}: ${val.toFixed(4)}`));
      svg.appendChild(rect);
    });

    svg.appendChild(
      svgEl(
        "text",
        {
          x: groupCenterX.toFixed(1),
          y: height - 24,
          "text-anchor": "middle",
          fill: "#cbd5e1",
          "font-size": "11.5",
          "font-weight": "600",
        },
        mKey
      )
    );
  });

  // Legend row
  let lx = pad.left;
  models.forEach((m, idx) => {
    svg.appendChild(
      svgEl("rect", {
        x: lx,
        y: height - 10,
        width: 10,
        height: 10,
        rx: 2,
        fill: colors[idx % colors.length],
      })
    );
    svg.appendChild(
      svgEl(
        "text",
        {
          x: lx + 14,
          y: height - 1,
          fill: "#94a3b8",
          "font-size": "11",
        },
        m.label
      )
    );
    lx += Math.max(115, m.label.length * 7.2 + 24);
  });

  return svg;
}
