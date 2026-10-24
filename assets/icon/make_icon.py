"""clairon app icon: a 'c' ring + speaker whose single sound wave closes the ring into a perfect circle.

All shapes are built with exact geometry (shapely) and written as SVG paths.
"""
import math
import sys
from pathlib import Path

import cairosvg
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

GOLD = "#E8A33D"
# ---- geometry (mark coordinates, centre at 0,0, y down like SVG) -------------
R_OUT = 400.0          # outer radius of the circle
T = 92.0               # ring / wave thickness
R_IN = R_OUT - T
GAP = 66.0             # width of the two cuts between 'c' and wave
ALPHA = math.radians(31)  # cone opening angle (each edge vs. horizontal)

RECT = (-200.0, -84.0, -112.0, 84.0)  # magnet: x0, y0, x1, y1
CONE_X0, CONE_H0 = RECT[2] + 0.0, 84.0
R_FRONT = R_IN - GAP   # cone front is an arc concentric with the circle
RES = 256


def ring():
    return Point(0, 0).buffer(R_OUT, resolution=RES).difference(Point(0, 0).buffer(R_IN, resolution=RES))


def wedge(offset: float) -> Polygon:
    """Right-opening wedge bounded by the extended cone edges, shifted by `offset` (+ = wider)."""
    big = 2000.0
    # upper edge passes (CONE_X0, -CONE_H0) with direction (cos a, -sin a); normal pointing into wedge:
    n_up = (math.sin(ALPHA), math.cos(ALPHA))
    n_lo = (math.sin(ALPHA), -math.cos(ALPHA))

    def halfplane(p, d, n, off):
        # points q with (q - p)·n >= -off
        px, py = p[0] - n[0] * off, p[1] - n[1] * off
        a = (px - d[0] * big, py - d[1] * big)
        b = (px + d[0] * big, py + d[1] * big)
        c = (b[0] + n[0] * big, b[1] + n[1] * big)
        e = (a[0] + n[0] * big, a[1] + n[1] * big)
        return Polygon([a, b, c, e])

    up = halfplane((CONE_X0, -CONE_H0), (math.cos(ALPHA), -math.sin(ALPHA)), n_up, offset)
    lo = halfplane((CONE_X0, CONE_H0), (math.cos(ALPHA), math.sin(ALPHA)), n_lo, offset)
    return up.intersection(lo)


def shapes():
    r = ring()
    # only cut on the right-hand side (the widened wedge's apex would otherwise nick the ring on the left)
    c_shape = r.difference(wedge(GAP / 2).intersection(box(0, -R_OUT * 2, R_OUT * 2, R_OUT * 2)))
    wave = r.intersection(wedge(-GAP / 2))
    magnet = box(*RECT)
    # flat front: largest x where the cone's corners still keep GAP distance to the ring
    x_front = CONE_X0
    while math.hypot(x_front + 1, CONE_H0 + math.tan(ALPHA) * (x_front + 1 - CONE_X0)) <= R_FRONT:
        x_front += 1
    cone = wedge(0).intersection(box(CONE_X0 - 1, -R_OUT, x_front, R_OUT))
    speaker = unary_union([magnet, cone])
    return c_shape, wave, speaker


def to_path(geom, s=1.0, dx=0.0, dy=0.0) -> str:
    polys = getattr(geom, "geoms", [geom])
    d = []
    for p in polys:
        for ring_ in [p.exterior, *p.interiors]:
            pts = list(ring_.coords)
            d.append("M" + " L".join(f"{x * s + dx:.2f},{y * s + dy:.2f}" for x, y in pts) + " Z")
    return " ".join(d)


def mark_svg(size=1024, color=GOLD, margin=0.06) -> str:
    s = size * (1 - 2 * margin) / (2 * R_OUT)
    c = size / 2
    paths = "".join(f'<path d="{to_path(g, s, c, c)}" fill="{color}" fill-rule="evenodd"/>' for g in shapes())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}">{paths}</svg>')


def app_icon_svg(size=1024, bg="#FFFFFF", color=GOLD, dark=False) -> str:
    """macOS Big Sur style: 824px rounded tile on a 1024 canvas, subtle shadow."""
    tile = size * 824 / 1024
    o = (size - tile) / 2
    rad = tile * 0.2237
    mark = tile * 0.70
    s = mark / (2 * R_OUT)
    c = size / 2
    paths = "".join(f'<path d="{to_path(g, s, c, c)}" fill="{color}" fill-rule="evenodd"/>' for g in shapes())
    top, bottom = ("#2B2724", "#1A1715") if dark else ("#FFFFFF", "#F3EEE6")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
<defs>
 <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bottom}"/></linearGradient>
 <filter id="sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="{size*0.012:.1f}" stdDeviation="{size*0.014:.1f}" flood-color="#000" flood-opacity="0.28"/></filter>
</defs>
<rect x="{o}" y="{o}" width="{tile}" height="{tile}" rx="{rad:.1f}" fill="url(#bg)" filter="url(#sh)"/>
{paths}
</svg>'''


if __name__ == "__main__":
    (OUT / "clairon-mark.svg").write_text(mark_svg())
    (OUT / "clairon-mark-white.svg").write_text(mark_svg(color="#FFFFFF"))
    (OUT / "clairon-appicon.svg").write_text(app_icon_svg())
    (OUT / "clairon-appicon-dark.svg").write_text(app_icon_svg(dark=True))
    for name in ["clairon-mark", "clairon-appicon", "clairon-appicon-dark"]:
        cairosvg.svg2png(url=str(OUT / f"{name}.svg"), write_to=str(OUT / f"{name}-1024.png"),
                         output_width=1024, output_height=1024)
    print("ok")
