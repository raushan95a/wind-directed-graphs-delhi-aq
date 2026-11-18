"""Renders the two .drawio diagrams to PDF with matplotlib (same geometry, colours and labels as the draw.io files).

Usage: python render_drawio.py   -> writes pdf/block_diagram.pdf and pdf/flow_diagram.pdf
"""
import os, re, html
import xml.etree.ElementTree as ET
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "pdf")
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.default": "regular"})


def style(s):
    d = {}
    for part in s.split(";"):
        if "=" in part: k, v = part.split("=", 1); d[k] = v
        elif part: d[part] = True
    return d


def lines_of(label):
    """HTML label -> list of (text, bold)."""
    label = label.replace("<sub>", "$_{").replace("</sub>", "}$")
    out = []
    for ln in re.split(r"<br\s*/?>", label):
        bold = "<b>" in ln
        txt = html.unescape(re.sub(r"<[^>]+>", "", ln)).strip()
        txt = txt.replace("$_{up}$", "$_{\\mathrm{up}}$")
        out.append((txt, bold))
    return out


def render(fn, out, FS, pad=12):
    root = ET.parse(os.path.join(HERE, fn)).getroot()
    cells = root.iter("mxCell")
    nodes, edges = {}, []
    for c in cells:
        st = style(c.get("style", ""))
        if c.get("vertex") == "1":
            g = c.find("mxGeometry")
            nodes[c.get("id")] = dict(x=float(g.get("x")), y=float(g.get("y")), w=float(g.get("width")),
                                      h=float(g.get("height")), st=st, label=c.get("value", ""))
        elif c.get("edge") == "1":
            edges.append(dict(a=c.get("source"), b=c.get("target"), st=st, label=c.get("value", "")))
    X0 = min(n["x"] for n in nodes.values()) - pad; X1 = max(n["x"] + n["w"] for n in nodes.values()) + pad + 36
    Y0 = min(n["y"] for n in nodes.values()) - pad; Y1 = max(n["y"] + n["h"] for n in nodes.values()) + pad
    W, H = X1 - X0, Y1 - Y0
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis("off")
    fig.canvas.draw(); rend = fig.canvas.get_renderer()

    def width(t, fs, b):
        a = ax.text(0, 0, t, fontsize=fs, fontweight="bold" if b else "normal")
        w = a.get_window_extent(rend).width * 72 / fig.dpi; a.remove(); return w

    def wrap(lines, fs, maxw):
        out = []
        for t, b in lines:
            words, cur = t.split(" "), ""
            for wd in words:
                cand = (cur + " " + wd).strip()
                if cur and width(cand, fs, b) > maxw: out.append((cur, b)); cur = wd
                else: cur = cand
            out.append((cur, b))
        return out

    def fit_text(n, x, y, lines, fs, color="#111111", maxw=None, maxh=None):
        maxw = maxw or n["w"] - 12; maxh = maxh or n["h"] - 6
        while True:
            wl = wrap(lines, fs, maxw); lh = fs * 1.2
            if lh * len(wl) <= maxh or fs <= 6: break
            fs -= 0.25
        y0 = y - lh * len(wl) / 2 + lh / 2
        for k, (t, b) in enumerate(wl):
            ax.text(x, y0 + k * lh, t, ha="center", va="center", fontsize=fs, color=color,
                    fontweight="bold" if b else "normal")

    for nid, n in nodes.items():
        st = n["st"]; x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        fc = st.get("fillColor", "#ffffff"); ec = st.get("strokeColor", "#1f3b57")
        fc = "none" if fc == "none" else fc
        if "rhombus" in st:
            ax.add_patch(Polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], closed=True,
                                 fc=fc, ec=ec, lw=1.0))
            fit_text(n, x + w / 2, y + h / 2, lines_of(n["label"]), FS, maxw=w * 0.6, maxh=h * 0.62)
        elif "ellipse" in st:
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, fc=fc, ec=ec, lw=1.0))
            fit_text(n, x + w / 2, y + h / 2, lines_of(n["label"]), FS + 1, color=st.get("fontColor", "#111111"))
        else:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=ec,
                                        lw=0.9, ls="--" if "dashed" in st else "-"))
            if st.get("verticalAlign") == "top":
                ax.text(x + 7, y + 9, html.unescape(n["label"]), ha="left", va="center", fontsize=9.5,
                        fontstyle="italic", color=st.get("fontColor", "#4a5560"))
            else:
                fit_text(n, x + w / 2, y + h / 2, lines_of(n["label"]), FS)

    def port(n, fx, fy): return (n["x"] + fx * n["w"], n["y"] + fy * n["h"])

    for e in edges:
        A, B, st = nodes[e["a"]], nodes[e["b"]], e["st"]
        acx, bcx = A["x"] + A["w"] / 2, B["x"] + B["w"] / 2
        if "exitX" in st:
            p0 = port(A, float(st["exitX"]), float(st["exitY"])); p1 = port(B, float(st["entryX"]), float(st["entryY"]))
            if float(st["exitX"]) == 1 and float(st["entryX"]) == 1:
                xo = max(p0[0], p1[0]) + 16; pts = [p0, (xo, p0[1]), (xo, p1[1]), p1]
            else:
                xm = (p0[0] + p1[0]) / 2; pts = [p0, (xm, p0[1]), (xm, p1[1]), p1]
        elif B["y"] >= A["y"] + A["h"] - 1:          # target below
            if B["x"] <= acx <= B["x"] + B["w"]: pts = [(acx, A["y"] + A["h"]), (acx, B["y"])]
            elif A["x"] <= bcx <= A["x"] + A["w"]: pts = [(bcx, A["y"] + A["h"]), (bcx, B["y"])]
            else:
                ym = (A["y"] + A["h"] + B["y"]) / 2
                pts = [(acx, A["y"] + A["h"]), (acx, ym), (bcx, ym), (bcx, B["y"])]
        else:                                          # side by side
            yc = B["y"] + B["h"] / 2
            pts = [(A["x"], yc), (B["x"] + B["w"], yc)] if bcx < acx else [(A["x"] + A["w"], yc), (B["x"], yc)]
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys, color="#333333", lw=0.9, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color="#333333", lw=0.9,
                                    shrinkA=0, shrinkB=0))
        if e["label"]:
            (xa, ya), (xb, yb) = (pts[1], pts[2]) if len(pts) == 4 else (pts[-2], pts[-1])
            if abs(xa - xb) < 1:  # vertical segment
                ax.text(xa + 4, (ya + yb) / 2, e["label"], fontsize=8.5, ha="left", va="center", color="#333333",
                        fontstyle="italic", bbox=dict(fc="white", ec="none", pad=0.5))
            else:
                ax.text((xa + xb) / 2, ya - 6, e["label"], fontsize=8.5, ha="center", va="center", color="#333333",
                        fontstyle="italic", bbox=dict(fc="white", ec="none", pad=0.5))
    fig.savefig(os.path.join(FIG, out)); plt.close(fig); print("wrote", out)


render("block_diagram.drawio", "block_diagram.pdf", 10)
render("flow_diagram.drawio", "flow_diagram.pdf", 10.5)
