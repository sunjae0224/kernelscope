// Build the v5 review memo (.docx) from memo_v5.js (generated from 중간보고서_v5_검토메모.md) with the report's styles.
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Footer, PageNumber, AlignmentType } = require("docx");
const L = require("./lib");
const body = require("./memo_v5");
const OUT = process.argv[2] || "memo.docx";
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: L.runFont })] })] });
const page = { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1418, right: 1418, footer: 700 } };
const doc = new Document({
  creator: "kernelscope", title: "중간보고서 v5 검토 메모",
  styles: { default: { document: { run: { font: L.runFont, size: L.BODY_PT }, paragraph: { spacing: { line: 360, lineRule: "auto" } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: L.runFont, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: L.runFont, size: 23, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: L.runFont, size: 21, bold: true, color: "000000" }, paragraph: { outlineLevel: 2 } },
    ] },
  numbering: { config: L.numberingConfig },
  sections: [{ properties: { page }, footers: { default: footer }, children: body() }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
