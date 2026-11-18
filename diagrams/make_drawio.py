"""Writes the two required draw.io diagrams (block diagram, working-flow diagram) as editable .drawio files."""
import os
from xml.sax.saxutils import escape

OUT = os.path.dirname(os.path.abspath(__file__))
BOX = "rounded=1;whiteSpace=wrap;html=1;fontSize=13;fontFamily=Helvetica;strokeColor=#1f3b57;"
STYLE = {
    "src": BOX + "fillColor=#e8eef5;",
    "proc": BOX + "fillColor=#ffffff;",
    "graph": BOX + "fillColor=#fff4e0;strokeColor=#9a5b00;",
    "learn": BOX + "fillColor=#e9f5ec;strokeColor=#2c6b3a;",
    "eval": BOX + "fillColor=#f3ecf7;strokeColor=#5b3a73;",
    "diag": BOX + "fillColor=#fdeceb;strokeColor=#8a2a24;",
    "dec": "rhombus;whiteSpace=wrap;html=1;fontSize=12;fontFamily=Helvetica;fillColor=#ffffff;strokeColor=#1f3b57;",
    "term": "ellipse;whiteSpace=wrap;html=1;fontSize=13;fontFamily=Helvetica;fillColor=#1f3b57;fontColor=#ffffff;strokeColor=#1f3b57;",
    "group": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fontStyle=2;verticalAlign=top;align=left;spacingLeft=6;dashed=1;fillColor=none;strokeColor=#7a8794;fontColor=#4a5560;",
}
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#333333;fontSize=9;"

class D:
    def __init__(s, name): s.name, s.cells, s.n = name, [], 2
    def node(s, label, x, y, w, h, kind):
        s.n += 1; cid = f"n{s.n}"
        s.cells.append(f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{STYLE[kind]}" vertex="1" parent="1">'
                       f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid
    def edge(s, a, b, label="", extra=""):
        s.n += 1
        s.cells.append(f'<mxCell id="e{s.n}" value="{escape(label)}" style="{EDGE}{extra}" edge="1" parent="1" source="{a}" target="{b}">'
                       f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    def save(s, fn):
        xml = ('<mxfile host="drawio"><diagram name="' + s.name + '" id="' + s.name + '"><mxGraphModel dx="800" dy="800" grid="1" '
               'gridSize="10" guides="1" page="1" pageScale="1" pageWidth="600" pageHeight="900" math="0" shadow="0"><root>'
               '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(s.cells) + "</root></mxGraphModel></diagram></mxfile>")
        open(os.path.join(OUT, fn), "w", encoding="utf-8").write(xml)

# ---------------- block diagram ----------------
b = D("block")
b.node("Data sources", 10, 5, 520, 95, "group")
s1 = b.node("<b>CPCB monitoring archive</b><br>hourly PM2.5, 39 Delhi stations,<br>2019&#8211;2024", 20, 25, 160, 65, "src")
s2 = b.node("<b>ERA5 reanalysis</b> (Open-Meteo)<br>10/100 m wind,<br>boundary-layer height", 190, 25, 160, 65, "src")
s3 = b.node("<b>Station catalogue</b><br>latitude / longitude<br>of each station", 360, 25, 160, 65, "src")
p = b.node("<b>Pre-processing and quality control</b><br>IST time fix &#183; hourly aggregation &#183; range / flat-line / PM10 checks &#183; station selection (34 kept)", 20, 120, 500, 50, "proc")
for s in (s1, s2, s3): b.edge(s, p)
dg = b.node("<b>Transport diagnostics</b><br>upwind/downwind lag asymmetry<br>transport ratio &#961; = s&#183;h / d<sub>up</sub>", 20, 195, 210, 70, "diag")
gc = b.node("<b>Graph construction</b> P(t) at forecast origin<br>none &#183; uniform &#183; distance &#183; correlation<br>static wind &#183; hourly wind &#183; ventilation-gated", 250, 195, 270, 70, "graph")
b.edge(p, dg); b.edge(p, gc)
ag = b.node("<b>Neighbour aggregation</b><br>m<sub>j</sub>(t&#8722;&#8467;) = &#931;<sub>i</sub> P<sub>ji</sub> x<sub>i</sub> / &#931;<sub>i</sub> P<sub>ji</sub> + edge mass", 250, 285, 270, 50, "graph")
b.edge(gc, ag)
l1 = b.node("<b>XGBoost</b> (fixed learner)<br>target: change from current value", 250, 355, 130, 60, "learn")
l2 = b.node("<b>Graph WaveNet-style GNN</b><br>forward-only diffusion", 390, 355, 130, 60, "learn")
b.edge(ag, l1); b.edge(gc, l2, extra="exitX=0.9;exitY=1;entryX=0.5;entryY=0;")
ev = b.node("<b>Evaluation</b><br>MAE by horizon (1&#8211;24 h) &#183; Diebold&#8211;Mariano + Holm &#183; week-block bootstrap<br>regimes: season &#183; ventilation &#183; transport ratio", 20, 440, 500, 60, "eval")
b.edge(l1, ev); b.edge(l2, ev); b.edge(dg, ev, "regime labels")
b.save("block_diagram.drawio")

# ---------------- working-flow diagram (two-column snake, compact) ----------------
f = D("flow")
LX, RX, W = 110, 370, 230
st = f.node("Start", LX + 65, 0, 100, 34, "term")
a1 = f.node("Download Delhi rows of the CPCB archive (2019&#8211;2024)", LX, 52, W, 44, "proc")
a2 = f.node("Hourly aggregation, IST fix, quality control", LX, 114, W, 44, "proc")
d1 = f.node("&#8805;75% complete in every training year?", LX + 25, 176, 180, 84, "dec")
dr = f.node("Exclude station", 0, 198, 90, 40, "proc")
a3 = f.node("Fetch ERA5 wind and boundary-layer height", LX, 278, W, 44, "proc")
a4 = f.node("Model-free lag-asymmetry test", LX, 340, W, 44, "diag")
a5 = f.node("Split by year; tune &#963; and gate slope on validation", RX, 0, W, 44, "proc")
a6 = f.node("Build graph P(t) at origin; aggregate neighbour features", RX, 62, W, 50, "graph")
a7 = f.node("Train learner on change from current value", RX, 130, W, 44, "learn")
a8 = f.node("Predict test year, horizons 1&#8211;24 h", RX, 192, W, 44, "learn")
d2 = f.node("More arms, horizons or splits?", RX + 25, 254, 180, 78, "dec")
a9 = f.node("DM + Holm tests, bootstrap, regime and transport-ratio analysis", RX, 350, W, 50, "eval")
en = f.node("End", RX + 65, 418, 100, 34, "term")
for x, y in [(st, a1), (a1, a2), (a2, d1), (a3, a4), (a5, a6), (a6, a7), (a7, a8), (a8, d2), (a9, en)]: f.edge(x, y)
f.edge(d1, a3, "yes"); f.edge(d1, dr, "no")
f.edge(a4, a5, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
f.edge(d2, a9, "no"); f.edge(d2, a6, "yes", "exitX=1;exitY=0.5;entryX=1;entryY=0.5;")
f.save("flow_diagram.drawio")
print("written")
