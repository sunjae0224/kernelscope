// Helpers that turn simple content blocks into docx-js elements for the midterm report.
const fs = require("fs");
const {
  Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, BorderStyle, VerticalAlign, PageBreak, LevelFormat, LineRuleType,
} = require("docx");

const FONT = process.env.REPORT_FONT || "맑은 고딕";
const MONO = process.env.REPORT_MONO || "Consolas";
const BODY_W = 9070; // A4 21.0 cm - 2 x 2.5 cm margins, in DXA
const BODY_PT = 20;  // half-points: 10 pt

const runFont = { ascii: FONT, eastAsia: FONT, hAnsi: FONT, cs: FONT };

// **bold** inline markup -> runs
function runs(text, opts = {}) {
  const out = [];
  String(text).split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (!part) return;
    const bold = part.startsWith("**") && part.endsWith("**");
    out.push(new TextRun({ text: bold ? part.slice(2, -2) : part, bold: bold || opts.bold, size: opts.size,
      font: opts.font || runFont, color: opts.color, italics: opts.italics }));
  });
  return out;
}

function p(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts),
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 200 },
    spacing: { after: opts.after ?? 100, line: opts.line ?? 360 },
    keepNext: opts.keepNext,
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(text, { size: 28, bold: true }),
    spacing: { before: 360, after: 160 }, keepNext: true });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(text, { size: 23, bold: true }),
    spacing: { before: 240, after: 100 }, keepNext: true });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(text, { size: 21, bold: true }),
    spacing: { before: 160, after: 60 }, keepNext: true });
}

function bullets(items, ref = "bullets") {
  return items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 340 } }));
}
function numbered(items, ref) {
  return bullets(items, ref);
}

function caption(text) {
  return new Paragraph({ children: runs(text, { size: 18, bold: false }), alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200, line: 300 } });
}

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// width in inches (<= 6.3); PNGs are rendered at 220 dpi
function figure(path, cap, widthIn) {
  const { w, h } = pngSize(path);
  const wIn = widthIn || Math.min(6.2, w / 220);
  const px = Math.round(wIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0, line: 240, lineRule: LineRuleType.AUTO }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: px, height: Math.round(px * h / w) } })] }),
    caption(cap),
  ];
}

function cell(text, width, o = {}) {
  const lines = String(text).split("\n");
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    columnSpan: o.span,
    verticalAlign: VerticalAlign.CENTER,
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: lines.map((line) => new Paragraph({ children: runs(line, { size: o.size || 17, bold: o.bold }),
      alignment: o.align || AlignmentType.CENTER, spacing: { after: 0, line: 280 } })),
  });
}

// widths: fractions or DXA; header: array; rows: arrays; align: per-column alignment list
function table(cap, widths, header, rows, o = {}) {
  const total = o.width || BODY_W;
  const sum = widths.reduce((a, b) => a + b, 0);
  const cols = widths.map((x) => Math.round(x / sum * total));
  cols[cols.length - 1] += total - cols.reduce((a, b) => a + b, 0);
  const aligns = o.align || cols.map(() => AlignmentType.CENTER);
  const trs = [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, cols[i], { fill: "E7ECF3", bold: true })) })];
  rows.forEach((r) => trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, cols[i], { align: aligns[i] })) })));
  return [
    new Paragraph({ children: runs(cap, { size: 18 }), alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: cols, rows: trs }),
    new Paragraph({ children: [], spacing: { after: 120 } }),
  ];
}

function code(lines) {
  return lines.map((line, i) => new Paragraph({
    children: [new TextRun({ text: line.length ? line : " ", font: { ascii: MONO, hAnsi: MONO, eastAsia: FONT, cs: MONO }, size: 15 })],
    shading: { type: ShadingType.CLEAR, fill: "F4F4F2", color: "auto" },
    spacing: { after: 0, line: 240 }, indent: { left: 120 }, keepLines: true,
  }));
}

function ref(n, text) {
  return new Paragraph({ children: runs(`[${n}] ${text}`, { size: 18 }), alignment: AlignmentType.LEFT,
    indent: { left: 440, hanging: 440 }, spacing: { after: 80, line: 300 } });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

const numberingConfig = [
  { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 440, hanging: 260 } } } }] },
  ...["n1", "n2", "n3", "n4", "n5", "n6"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL,
    text: "%1)", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 300 } } } }] })),
];

module.exports = { FONT, BODY_W, BODY_PT, runFont, runs, p, h1, h2, h3, bullets, numbered, caption, figure, table, cell,
  code, ref, pageBreak, numberingConfig, BorderStyle, AlignmentType };
