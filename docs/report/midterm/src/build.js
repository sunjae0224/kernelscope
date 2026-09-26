// Build the midterm report .docx following the school template (붙임2).
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  BorderStyle, VerticalAlign, Footer, PageNumber, HeightRule,
} = require("docx");
const L = require("./lib");
const body1 = require("./content1");
const body2 = require("./content2");

const OUT = process.argv[2] || "report.docx";
const TITLE = "GPU 프로파일링·시뮬레이션 기반 LLM 추론 attention 커널 동적 선택 시스템";
const GITHUB = "https://github.com/sunjae0224/kernelscope";
const GRAY = "D9D9D9";

function t(text, o = {}) {
  return new Paragraph({ alignment: o.align || AlignmentType.CENTER, spacing: { before: o.before || 0, after: o.after || 0, line: o.line || 300 },
    children: L.runs(text, { size: o.size || 20, bold: o.bold }), border: o.border });
}
function c(children, width, o = {}) {
  return new TableCell({ width: { size: width, type: WidthType.DXA }, columnSpan: o.span, rowSpan: o.rowSpan,
    verticalAlign: o.valign || VerticalAlign.CENTER, margins: { top: 80, bottom: 80, left: 120, right: 120 },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    children: Array.isArray(children) ? children : [children] });
}

// ---- cover: evaluation table (평가표는 수정하지 않습니다) ----
const W = [1300, 4300, 3470];
const cover = [
  new Paragraph({ children: L.runs("붙임2", { size: 22, bold: true }), spacing: { after: 600 },
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 2 }, bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 2 },
      left: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 4 }, right: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 4 } },
    indent: { right: 8200 } }),
  t("연구논문/작품 중간보고서", { size: 44, before: 600, after: 200 }),
  new Paragraph({ children: [], spacing: { after: 700 }, border: { bottom: { style: BorderStyle.SINGLE, size: 36, color: "8C8C8C", space: 1 } } }),
  t("2026 학년도 제 2 학기", { size: 26, after: 700 }),
  new Table({ width: { size: 9070, type: WidthType.DXA }, columnWidths: W, rows: [
    new TableRow({ height: { value: 1100, rule: HeightRule.ATLEAST }, children: [
      c(t("제목", { bold: true }), W[0], { fill: GRAY }),
      c(t(TITLE, { size: 19, align: AlignmentType.LEFT, line: 320 }), W[1], { fill: GRAY }),
      c([t("○ 논문(   ) 작품( ✓ )", { size: 19, align: AlignmentType.LEFT }), t("※해당란 체크", { size: 17, align: AlignmentType.LEFT, before: 80 })], W[2], { fill: GRAY }),
    ] }),
    new TableRow({ height: { value: 800, rule: HeightRule.ATLEAST }, children: [
      c(t("GitHub\nURL".split("\n").join(" "), { bold: true }), W[0], { fill: GRAY }),
      c(t(`${GITHUB} (design-1-3 브랜치)`, { size: 19, align: AlignmentType.LEFT }), W[1] + W[2], { span: 2, fill: GRAY }),
    ] }),
    new TableRow({ children: [
      c(t("평가등급", { bold: true }), W[0], { fill: GRAY }),
      c(t("지도교수 수정보완 사항", { bold: true }), W[1], { fill: GRAY }),
      c(t("팀원 명단", { bold: true }), W[2], { fill: GRAY }),
    ] }),
    new TableRow({ height: { value: 3000, rule: HeightRule.ATLEAST }, children: [
      c([t("A,B,F중", { size: 19 }), t("택1", { size: 19 }), t("(지도교수가", { size: 15, before: 60 }), t("부여)", { size: 15 })], W[0]),
      c([t("○", { align: AlignmentType.LEFT, after: 600 }), t("○", { align: AlignmentType.LEFT, after: 600 }), t("○", { align: AlignmentType.LEFT })], W[1], { valign: VerticalAlign.TOP }),
      c([t("이 선 재 (인) (학번:            )", { size: 19 })], W[2]),
    ] }),
  ] }),
];

// ---- second page: date and signature, guidance box removed ----
const signature = [
  t("2026 년   9 월   27 일", { size: 26, before: 2400, after: 1400 }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [
    ...L.runs("지도교수 : o o o     ", { size: 26 }),
    new TextRun({ text: "서명                        ", size: 26, underline: {}, font: L.runFont }),
  ] }),
];

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: L.runFont })] })] });

const page = { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1418, right: 1418, footer: 700 } };

const doc = new Document({
  creator: "이선재",
  title: TITLE,
  styles: {
    default: { document: { run: { font: L.runFont, size: L.BODY_PT }, paragraph: { spacing: { line: 360, lineRule: "auto" } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: L.runFont, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: L.runFont, size: 23, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: L.runFont, size: 21, bold: true, color: "000000" }, paragraph: { outlineLevel: 2 } },
    ],
  },
  numbering: { config: L.numberingConfig },
  sections: [
    { properties: { page }, children: [...cover, L.pageBreak(), ...signature] },
    { properties: { page: { ...page, pageNumbers: { start: 1 } } }, footers: { default: footer }, children: [...body1(), ...body2()] },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
