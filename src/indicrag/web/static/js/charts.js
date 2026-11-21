/**
 * Vector Chart Renderers for IndicRAG-QA Evaluation Lab & Overview.
 * High-performance, resolution-independent SVG generators.
 */

export const Charts = {
  /**
   * Render Systems Ladder Bar Comparison (Clean vs. Perturbed).
   */
  renderSystemsLadder(containerId, data) {
    const el = document.getElementById(containerId);
    if (!el || !data || data.length === 0) return;

    // Filter Dev Clean and Perturbed rows
    const cleanRows = data.filter(d => d.corpus === "clean" || !d.corpus);
    const pertRows = data.filter(d => d.corpus === "perturbed");

    const systems = ["S0", "S1", "S2", "S3", "S4", "S5"];
    const width = 640;
    const height = 260;
    const padding = { top: 25, right: 30, bottom: 40, left: 50 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const groupW = chartW / systems.length;
    const barW = groupW * 0.35;

    let barsSvg = "";

    systems.forEach((sys, idx) => {
      const clean = cleanRows.find(r => r.system === sys) || { acc_ans: 0 };
      const pert = pertRows.find(r => r.system === sys) || { acc_ans: 0 };

      const cleanVal = clean.acc_ans || 0;
      const pertVal = pert.acc_ans || 0;

      const cleanH = (cleanVal / 100) * chartH;
      const pertH = (pertVal / 100) * chartH;

      const groupX = padding.left + idx * groupW;
      const cleanX = groupX + groupW * 0.1;
      const pertX = cleanX + barW + 4;

      const cleanY = padding.top + chartH - cleanH;
      const pertY = padding.top + chartH - pertH;

      barsSvg += `
        <!-- ${sys} Clean -->
        <rect x="${cleanX}" y="${cleanY}" width="${barW}" height="${cleanH}" rx="3" fill="#38bdf8" fill-opacity="0.9">
          <title>${sys} Clean: ${cleanVal.toFixed(1)}%</title>
        </rect>
        <text x="${cleanX + barW / 2}" y="${cleanY - 5}" font-family="IBM Plex Mono" font-size="10" fill="#38bdf8" text-anchor="middle">${cleanVal.toFixed(0)}%</text>

        <!-- ${sys} Perturbed -->
        <rect x="${pertX}" y="${pertY}" width="${barW}" height="${pertH}" rx="3" fill="#f43f5e" fill-opacity="0.8">
          <title>${sys} Perturbed: ${pertVal.toFixed(1)}%</title>
        </rect>
        <text x="${pertX + barW / 2}" y="${pertY - 5}" font-family="IBM Plex Mono" font-size="10" fill="#f43f5e" text-anchor="middle">${pertVal.toFixed(0)}%</text>

        <!-- X Label -->
        <text x="${groupX + groupW / 2}" y="${height - 12}" font-family="Rajdhani" font-size="13" font-weight="700" fill="#cbd5e1" text-anchor="middle">${sys}</text>
      `;
    });

    // Grid lines
    let gridSvg = "";
    for (let pct = 0; pct <= 100; pct += 25) {
      const y = padding.top + chartH - (pct / 100) * chartH;
      gridSvg += `
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="rgba(148, 163, 184, 0.15)" stroke-dasharray="3,3" />
        <text x="${padding.left - 8}" y="${y + 4}" font-family="IBM Plex Mono" font-size="10" fill="#64748b" text-anchor="end">${pct}%</text>
      `;
    }

    el.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:100%;">
        ${gridSvg}
        ${barsSvg}
      </svg>
    `;
  },

  /**
   * Render Cross-Script Hinglish Ablation Bar Chart.
   */
  renderHinglishAblation(containerId, data) {
    const el = document.getElementById(containerId);
    if (!el || !data) return;

    const width = 500;
    const height = 220;
    const padding = { top: 30, right: 30, bottom: 40, left: 160 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const rowH = chartH / data.length;
    const barH = rowH * 0.45;

    let rowsSvg = "";

    data.forEach((item, idx) => {
      const val = item.acc_ans || 0;
      const bW = Math.max((val / 100) * chartW, 3); // min visible bar
      const y = padding.top + idx * rowH + (rowH - barH) / 2;

      rowsSvg += `
        <text x="${padding.left - 12}" y="${y + barH / 2 + 4}" font-family="Rajdhani" font-size="12" font-weight="600" fill="#cbd5e1" text-anchor="end">
          ${item.method_name || item.method}
        </text>
        <rect x="${padding.left}" y="${y}" width="${bW}" height="${barH}" rx="3" fill="${val > 0 ? '#38bdf8' : '#f43f5e'}" fill-opacity="0.85">
          <title>${item.method}: ${val.toFixed(2)}%</title>
        </rect>
        <text x="${padding.left + bW + 8}" y="${y + barH / 2 + 4}" font-family="IBM Plex Mono" font-size="11" fill="#fff" font-weight="600">
          ${val.toFixed(2)}% (FRR: ${(item.false_refusal_rate || item.frr).toFixed(1)}%)
        </text>
      `;
    });

    el.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:100%;">
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="rgba(148, 163, 184, 0.3)" />
        ${rowsSvg}
      </svg>
    `;
  },
};
