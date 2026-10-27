"""Markdown (headings, paragraphs, lists, pipe tables, **bold**) -> a docx-js content module using the report's lib.js."""
import json, re, sys
src = open(sys.argv[1], encoding="utf-8").read().splitlines()
items = []
def js(s): return json.dumps(s, ensure_ascii=False)
def strip_code(t): return re.sub(r"`(.+?)`", r"\1", t)
i = 0; para = []; lst = []
def flush():
    global para, lst
    if para: items.append("L.p(%s)" % js(strip_code(" ".join(para)))); para = []
    if lst: items.append("...L.bullets([%s])" % ", ".join(js(strip_code(x)) for x in lst)); lst = []
while i < len(src):
    line = src[i]
    if line.startswith("|") and i + 1 < len(src) and re.match(r"^\|[\s:|-]+\|$", src[i+1]):
        flush(); hdr = [c.strip() for c in line.strip("|").split("|")]; rows = []; i += 2
        while i < len(src) and src[i].startswith("|"):
            rows.append([strip_code(c.strip()) for c in src[i].strip("|").split("|")]); i += 1
        n = len(hdr)
        widths = [1.0] * n
        if n == 6: widths = [0.5, 3.2, 0.9, 1.4, 0.8, 3.2]
        items.append("...L.table(%s, %s, %s, %s, { align: [%s] })" % (js(""), json.dumps(widths), json.dumps(hdr, ensure_ascii=False),
                     json.dumps(rows, ensure_ascii=False), ", ".join("LEFT" for _ in hdr)))
        continue
    m = re.match(r"^(#{1,3})\s+(.*)", line)
    if m:
        flush(); lvl = len(m.group(1)); fn = {1: "L.h1", 2: "L.h2", 3: "L.h3"}[lvl]
        items.append("%s(%s)" % (fn, js(strip_code(m.group(2))))); i += 1; continue
    if line.strip() == "---": flush(); i += 1; continue
    m = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)", line)
    if m:
        if para: items.append("L.p(%s)" % js(strip_code(" ".join(para)))); para = []
        lst.append(m.group(1)); i += 1; continue
    if not line.strip(): flush(); i += 1; continue
    if lst and line.startswith("  "): lst[-1] += " " + line.strip(); i += 1; continue
    if lst: items.append("...L.bullets([%s])" % ", ".join(js(strip_code(x)) for x in lst)); lst = []
    para.append(line.strip()); i += 1
flush()
out = 'const L = require("%s");\nconst LEFT = L.AlignmentType.LEFT;\nmodule.exports = () => [\n  %s,\n];\n' % (sys.argv[3], ",\n  ".join(items))
open(sys.argv[2], "w", encoding="utf-8").write(out)
print(len(items), "blocks")
