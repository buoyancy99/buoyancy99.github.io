/*
  Research section renderer.

  This keeps the existing CSS structure:
  - `.paper-thumbnail`
  - `.paper-info`
  - custom `<heading>` tag used by your CSS
  - uses existing `toggleblock()` + `togglebib()` from `js/hidebib.js`
*/

(function () {
  "use strict";

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function normalizePaper(p) {
    if (!p || !p.id) throw new Error("Invalid paper: missing id");
    const absId = `${p.id}_abs`;
    return { ...p, absId };
  }

  function buildLinksHtml(paper) {
    if (!paper.links || paper.links.length === 0) return "";

    const parts = paper.links.map((l) => {
      const label = escapeHtml(l.label);

      if (l.onClick === "abstract") {
        return `<a href="javascript:toggleblock('${paper.absId}')">${label}</a>`;
      }
      if (l.onClick === "bibtex") {
        return `<a shape="rect" href="javascript:togglebib('${paper.id}')" class="togglebib">${label}</a>`;
      }

      const href = escapeHtml(l.href);
      const isExternal = /^https?:\/\//i.test(l.href);
      const extra = isExternal ? ` target="_blank" rel="noopener noreferrer"` : "";
      return `<a href="${href}"${extra}>${label}</a>`;
    });

    return parts.join(" | ");
  }

  function paperToHtml(p) {
    const paper = normalizePaper(p);

    const thumbHref = paper.thumbnail.href || paper.website || "#";
    const thumbIsExternal = /^https?:\/\//i.test(thumbHref);
    const thumbExtra = thumbIsExternal
      ? ` target="_blank" rel="noopener noreferrer"`
      : "";

    const titleHref = paper.website || thumbHref || "#";
    const titleIsExternal = /^https?:\/\//i.test(titleHref);
    const titleExtra = titleIsExternal
      ? ` target="_blank" rel="noopener noreferrer"`
      : "";

    const noteLine = paper.noteHtml ? `${paper.noteHtml}<br />` : "";
    const linksLine = buildLinksHtml(paper);
    const linksBlock = linksLine ? `${linksLine}<br />` : "";

    // Note: `authorsHtml`, `venueHtml`, `abstractHtml`, `bibtex` are treated as trusted,
    // since they're authored by you. Everything else is escaped.
    return `
      <div class="row research-paper-row">
        <div class="col-md-4">
          <div class="paper-thumbnail">
            <a href="${escapeHtml(thumbHref)}"${thumbExtra}>
              <img
                loading="lazy"
                src="${escapeHtml(paper.thumbnail.src)}"
                alt="${escapeHtml(paper.thumbnail.alt || "paper thumbnail")}"
                width="${Number(paper.thumbnail.width || 504)}"
                height="${Number(paper.thumbnail.height || 300)}" />
            </a>
          </div>
        </div>
        <div class="col-md-8">
          <div class="paper-info" id="${escapeHtml(paper.id)}">
            <a href="${escapeHtml(titleHref)}"${titleExtra}>
              <heading>${escapeHtml(paper.title)}</heading>
            </a>
            <br />
            ${paper.authorsHtml}
            <br />
            ${noteLine}
            ${paper.venueHtml}
            <br />
            <br />
            ${linksBlock}
            <pre xml:space="preserve" style="display: none">${escapeHtml(
              paper.bibtex || ""
            )}</pre>
          </div>
          <p style="text-align: justify">
            <i id="${escapeHtml(paper.absId)}" style="display: none">${paper.abstractHtml}</i>
          </p>
        </div>
      </div>
    `.trim();
  }

  function renderResearchPapersInto(container, papers) {
    if (!container) return;
    const list = Array.isArray(papers) ? papers : [];
    container.innerHTML = list.map(paperToHtml).join("\n\n");
  }

  /**
   * Public entrypoint. Called from `index.html`.
   */
  window.renderResearchPapers = function renderResearchPapers() {
    const mount = document.getElementById("research-papers");
    renderResearchPapersInto(mount, window.RESEARCH_PAPERS);
  };
})();

