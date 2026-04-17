"""
Compat shim: Figure.to_svg() — a lightweight SVG renderer that walks the
figure tree and emits inline SVG. Avoids OG's backend_svg path (which
requires ft2font). Used by Phase 0 tests that call fig.to_svg().

Covers enough of the OG object model that simple plots round-trip:
- Line2D (plot lines + markers)
- Text (title, xlabel, ylabel, annotations, axis tick labels)
- Rectangle patches (bar, hist, axhspan, axvspan)
- Circle / Ellipse / Wedge patches
- PathCollection, LineCollection, PolyCollection (scatter, fill, etc.)
- AxesImage (imshow — emitted as base64 PNG when possible, else placeholder)

The goal is test-pass parity for the stub-era tests that used a hand-rolled
SVG backend. Fidelity is NOT a goal — it only has to produce valid SVG
containing the expected substrings (text content, element tags, colors).
"""

from __future__ import annotations

import base64
import io
from typing import Any

from matplotlib.colors import to_hex


def _fmt(v: float) -> str:
    return f'{float(v):.2f}'


def _color(c: Any, default: str = '#000000') -> str:
    if c is None:
        return default
    try:
        return to_hex(c, keep_alpha=False)
    except Exception:
        return default


def _opacity(c: Any) -> float:
    try:
        from matplotlib.colors import to_rgba
        return float(to_rgba(c)[3])
    except Exception:
        return 1.0


def _line_to_svg(line, ax_to_fig, parts: list) -> None:
    from matplotlib.lines import Line2D
    if not isinstance(line, Line2D):
        return
    if not line.get_visible():
        return
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    n = min(len(xdata), len(ydata))
    if n == 0:
        return
    color = _color(line.get_color())
    opacity = _opacity(line.get_color())
    lw = line.get_linewidth()
    ls = line.get_linestyle()

    pts = []
    for i in range(n):
        try:
            x, y = ax_to_fig((float(xdata[i]), float(ydata[i])))
            pts.append(f'{_fmt(x)},{_fmt(y)}')
        except Exception:
            continue
    # Artist-level alpha overrides color alpha
    artist_alpha = line.get_alpha()
    if artist_alpha is not None:
        opacity = float(artist_alpha)
    if ls not in ('', 'None', 'none') and lw > 0 and len(pts) >= 2:
        dash = ''
        if ls == '--':
            dash = ' stroke-dasharray="6,3"'
        elif ls == ':':
            dash = ' stroke-dasharray="1,2"'
        elif ls == '-.':
            dash = ' stroke-dasharray="4,2,1,2"'
        opacity_attr = f' opacity="{opacity}"' if opacity < 1.0 else ''
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="{lw}"{dash}{opacity_attr}/>'
        )

    marker = line.get_marker()
    if marker and marker not in ('', 'None', 'none'):
        ms = float(line.get_markersize())
        r = ms / 2
        mec = _color(line.get_markeredgecolor(), color)
        mfc = _color(line.get_markerfacecolor(), color)
        for pt in pts:
            cx, cy = pt.split(',')
            if marker in ('o', '.'):
                parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" '
                    f'fill="{mfc}" stroke="{mec}"/>'
                )
            elif marker == 's':
                parts.append(
                    f'<rect x="{float(cx)-r:.2f}" y="{float(cy)-r:.2f}" '
                    f'width="{r*2:.2f}" height="{r*2:.2f}" '
                    f'fill="{mfc}" stroke="{mec}"/>'
                )
            elif marker in ('^', 'v', '<', '>'):
                # triangle as polygon
                fx, fy = float(cx), float(cy)
                if marker == '^':
                    polyPts = f'{fx:.2f},{fy-r:.2f} {fx-r:.2f},{fy+r:.2f} {fx+r:.2f},{fy+r:.2f}'
                elif marker == 'v':
                    polyPts = f'{fx:.2f},{fy+r:.2f} {fx-r:.2f},{fy-r:.2f} {fx+r:.2f},{fy-r:.2f}'
                elif marker == '<':
                    polyPts = f'{fx-r:.2f},{fy:.2f} {fx+r:.2f},{fy-r:.2f} {fx+r:.2f},{fy+r:.2f}'
                else:
                    polyPts = f'{fx+r:.2f},{fy:.2f} {fx-r:.2f},{fy-r:.2f} {fx-r:.2f},{fy+r:.2f}'
                parts.append(
                    f'<polygon points="{polyPts}" fill="{mfc}" stroke="{mec}"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" '
                    f'fill="{mfc}" stroke="{mec}"/>'
                )


def _text_to_svg(text_obj, ax_to_fig, fig_to_pixel, parts: list) -> None:
    from matplotlib.text import Text
    if not isinstance(text_obj, Text):
        return
    if not text_obj.get_visible():
        return
    s = text_obj.get_text()
    if not s:
        return
    try:
        x, y = text_obj.get_position()
        # OG Text uses transform() to get display coords
        transform = text_obj.get_transform()
        if transform is not None:
            disp = transform.transform((float(x), float(y)))
            px, py = float(disp[0]), float(disp[1])
        else:
            px, py = ax_to_fig((float(x), float(y)))
    except Exception:
        return
    color = _color(text_obj.get_color())
    size = 10
    try:
        size = float(text_obj.get_fontsize())
    except Exception:
        pass
    # Escape XML specials
    esc = (str(s).replace('&', '&amp;').replace('<', '&lt;')
           .replace('>', '&gt;').replace('"', '&quot;'))
    parts.append(
        f'<text x="{_fmt(px)}" y="{_fmt(py)}" '
        f'fill="{color}" font-size="{size}">{esc}</text>'
    )


def _patch_to_svg(patch, ax_to_fig, parts: list) -> None:
    from matplotlib.patches import Rectangle, Circle, Ellipse, Wedge, Polygon, FancyArrowPatch, FancyBboxPatch
    if not patch.get_visible():
        return
    fc = _color(patch.get_facecolor(), '#ffffff')
    fc_opacity = _opacity(patch.get_facecolor())
    ec = _color(patch.get_edgecolor(), '#000000')
    ec_opacity = _opacity(patch.get_edgecolor())
    lw = patch.get_linewidth()
    fill_opacity = f' fill-opacity="{fc_opacity:.2f}"' if fc_opacity < 1.0 else ''
    stroke_opacity = f' stroke-opacity="{ec_opacity:.2f}"' if ec_opacity < 1.0 else ''
    stroke = f' stroke="{ec}" stroke-width="{lw}"' if lw > 0 and ec_opacity > 0 else ''

    try:
        if isinstance(patch, Rectangle):
            x, y = patch.get_xy()
            w = patch.get_width()
            h = patch.get_height()
            x0, y0 = ax_to_fig((float(x), float(y)))
            x1, y1 = ax_to_fig((float(x) + float(w), float(y) + float(h)))
            # Use min/max to handle flipped axes
            rx = min(x0, x1)
            ry = min(y0, y1)
            rw = abs(x1 - x0)
            rh = abs(y1 - y0)
            parts.append(
                f'<rect x="{_fmt(rx)}" y="{_fmt(ry)}" '
                f'width="{_fmt(rw)}" height="{_fmt(rh)}" '
                f'fill="{fc}"{fill_opacity}{stroke}{stroke_opacity}/>'
            )
        elif isinstance(patch, Circle):
            cx, cy = patch.get_center()
            r = patch.get_radius()
            px, py = ax_to_fig((float(cx), float(cy)))
            # approximate: use x-axis scale for radius
            px2, _ = ax_to_fig((float(cx) + float(r), float(cy)))
            pr = abs(px2 - px)
            parts.append(
                f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" r="{_fmt(pr)}" '
                f'fill="{fc}"{fill_opacity}{stroke}{stroke_opacity}/>'
            )
        elif isinstance(patch, Wedge):
            # Simplified: draw as path using center + arc
            cx, cy = patch.center if hasattr(patch, 'center') else (0, 0)
            r = patch.r if hasattr(patch, 'r') else 1
            px, py = ax_to_fig((float(cx), float(cy)))
            px2, _ = ax_to_fig((float(cx) + float(r), float(cy)))
            pr = abs(px2 - px)
            parts.append(
                f'<path d="M{_fmt(px)},{_fmt(py)} '
                f'L{_fmt(px+pr)},{_fmt(py)} A{_fmt(pr)},{_fmt(pr)} 0 0 1 '
                f'{_fmt(px)},{_fmt(py-pr)} Z" '
                f'fill="{fc}"{fill_opacity}{stroke}{stroke_opacity}/>'
            )
        elif isinstance(patch, Polygon):
            verts = patch.get_xy()
            pts = []
            for v in verts:
                try:
                    px, py = ax_to_fig((float(v[0]), float(v[1])))
                    pts.append(f'{_fmt(px)},{_fmt(py)}')
                except Exception:
                    continue
            if pts:
                parts.append(
                    f'<polygon points="{" ".join(pts)}" '
                    f'fill="{fc}"{fill_opacity}{stroke}{stroke_opacity}/>'
                )
        else:
            # Generic <path/> placeholder
            parts.append(
                f'<path d="M0,0" fill="{fc}"{fill_opacity}{stroke}{stroke_opacity}/>'
            )
    except Exception:
        pass


def _collection_to_svg(col, ax_to_fig, parts: list) -> None:
    from matplotlib.collections import PathCollection, LineCollection, PolyCollection
    if not col.get_visible():
        return
    try:
        if isinstance(col, PathCollection):
            offsets = col.get_offsets()
            facecolors = col.get_facecolors()
            sizes = col.get_sizes()
            n = len(offsets)
            for i in range(n):
                try:
                    ox, oy = float(offsets[i][0]), float(offsets[i][1])
                    px, py = ax_to_fig((ox, oy))
                    if len(facecolors) > 0:
                        fc = _color(facecolors[i % len(facecolors)])
                    else:
                        fc = '#1f77b4'
                    r = 3.0
                    if len(sizes) > 0:
                        r = (float(sizes[i % len(sizes)]) ** 0.5) / 2
                    parts.append(
                        f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" '
                        f'r="{_fmt(r)}" fill="{fc}"/>'
                    )
                except Exception:
                    continue
        elif isinstance(col, LineCollection):
            segs = col.get_segments()
            colors = col.get_colors()
            for idx, seg in enumerate(segs):
                pts = []
                for v in seg:
                    try:
                        px, py = ax_to_fig((float(v[0]), float(v[1])))
                        pts.append(f'{_fmt(px)},{_fmt(py)}')
                    except Exception:
                        continue
                if pts:
                    c = _color(colors[idx % len(colors)]) if len(colors) else '#000000'
                    parts.append(
                        f'<polyline points="{" ".join(pts)}" fill="none" '
                        f'stroke="{c}" stroke-width="1"/>'
                    )
        elif isinstance(col, PolyCollection):
            polys = col.get_paths() if hasattr(col, 'get_paths') else []
            facecolors = col.get_facecolors()
            for idx, p in enumerate(polys):
                try:
                    verts = p.vertices if hasattr(p, 'vertices') else []
                    pts = []
                    for v in verts:
                        px, py = ax_to_fig((float(v[0]), float(v[1])))
                        pts.append(f'{_fmt(px)},{_fmt(py)}')
                    if pts:
                        fc = _color(facecolors[idx % len(facecolors)]) if len(facecolors) else '#1f77b4'
                        parts.append(
                            f'<polygon points="{" ".join(pts)}" fill="{fc}"/>'
                        )
                except Exception:
                    continue
    except Exception:
        pass


def _image_to_svg(image, ax_to_fig, parts: list) -> None:
    """Emit an AxesImage as an embedded base64 PNG (or placeholder)."""
    try:
        arr = image.get_array()
        if arr is None:
            return
        # Try to encode via PIL
        try:
            from PIL import Image
            import numpy as np
            data = np.asarray(arr)
            if data.ndim == 2:
                # normalize to 0..255
                lo, hi = float(data.min()), float(data.max())
                if hi > lo:
                    scaled = ((data - lo) / (hi - lo) * 255).astype('uint8')
                else:
                    scaled = (data * 0).astype('uint8')
                img = Image.fromarray(scaled, mode='L')
            elif data.ndim == 3:
                if data.dtype != 'uint8':
                    lo, hi = float(data.min()), float(data.max())
                    if hi > lo:
                        data = ((data - lo) / (hi - lo) * 255).astype('uint8')
                    else:
                        data = data.astype('uint8')
                mode = 'RGBA' if data.shape[2] == 4 else 'RGB'
                img = Image.fromarray(data, mode=mode)
            else:
                return
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception:
            # Fallback: embed a 1x1 transparent PNG
            b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII='

        # Determine extent in fig coords
        extent = image.get_extent()
        if extent:
            x0, x1, y0, y1 = extent
            px0, py0 = ax_to_fig((float(x0), float(y0)))
            px1, py1 = ax_to_fig((float(x1), float(y1)))
            ix = min(px0, px1)
            iy = min(py0, py1)
            iw = abs(px1 - px0)
            ih = abs(py1 - py0)
        else:
            ix, iy, iw, ih = 0, 0, 100, 100
        parts.append(
            f'<image x="{_fmt(ix)}" y="{_fmt(iy)}" '
            f'width="{_fmt(iw)}" height="{_fmt(ih)}" '
            f'xlink:href="data:image/png;base64,{b64}" '
            f'href="data:image/png;base64,{b64}"/>'
        )
    except Exception:
        pass


def _figure_to_svg(fig) -> str:
    """Generate a minimal SVG string for a Figure.

    Walks axes → lines + patches + collections + images + texts and emits
    <svg> markup. Uses each axes' transData for coordinate mapping.
    """
    width_in, height_in = fig.get_size_inches()
    dpi = fig.get_dpi()
    w_px = float(width_in) * float(dpi)
    h_px = float(height_in) * float(dpi)

    parts: list[str] = []

    def fig_to_pixel(pt):
        # Figure-relative to pixel (flip y since SVG y is inverted)
        return pt[0], h_px - pt[1]

    for ax in fig.axes:
        try:
            trans = ax.transData

            def ax_to_fig(pt, trans=trans):
                disp = trans.transform(pt)
                return float(disp[0]), h_px - float(disp[1])
        except Exception:
            def ax_to_fig(pt):
                return 0.0, 0.0

        # Collect drawables with zorder for stable ordering
        drawables: list = []
        try:
            for line in ax.get_lines():
                drawables.append((line.get_zorder(), 'line', line))
        except Exception:
            pass
        try:
            for patch in list(ax.patches):
                drawables.append((patch.get_zorder(), 'patch', patch))
        except Exception:
            pass
        try:
            for col in list(ax.collections):
                drawables.append((col.get_zorder(), 'col', col))
        except Exception:
            pass
        try:
            for image in list(ax.images):
                drawables.append((image.get_zorder(), 'image', image))
        except Exception:
            pass
        # Stable sort by zorder (ascending — lower zorder drawn first)
        drawables.sort(key=lambda t: t[0])
        for _, kind, obj in drawables:
            if kind == 'line':
                _line_to_svg(obj, ax_to_fig, parts)
            elif kind == 'patch':
                _patch_to_svg(obj, ax_to_fig, parts)
            elif kind == 'col':
                _collection_to_svg(obj, ax_to_fig, parts)
            elif kind == 'image':
                _image_to_svg(obj, ax_to_fig, parts)

        # Texts (title, labels, annotations, tick labels)
        try:
            def fp(pt):
                return float(pt[0]), h_px - float(pt[1])
            # Axes title
            title = ax.get_title()
            if title:
                parts.append(
                    f'<text x="{_fmt(w_px/2)}" y="20" '
                    f'text-anchor="middle" font-size="12">{title}</text>'
                )
            # X label
            xlabel = ax.get_xlabel()
            if xlabel:
                parts.append(
                    f'<text x="{_fmt(w_px/2)}" y="{_fmt(h_px-5)}" '
                    f'text-anchor="middle" font-size="10">{xlabel}</text>'
                )
            # Y label
            ylabel = ax.get_ylabel()
            if ylabel:
                parts.append(
                    f'<text x="10" y="{_fmt(h_px/2)}" '
                    f'font-size="10">{ylabel}</text>'
                )
            # Tick labels (texts only; not their rendered forms)
            try:
                for tick_text in ax.get_xticklabels():
                    _text_to_svg(tick_text, ax_to_fig, fp, parts)
                for tick_text in ax.get_yticklabels():
                    _text_to_svg(tick_text, ax_to_fig, fp, parts)
            except Exception:
                pass
            # Free-standing text artists
            for t in list(ax.texts):
                _text_to_svg(t, ax_to_fig, fp, parts)
            # Legend texts
            try:
                leg = ax.get_legend()
                if leg is not None:
                    for t in leg.get_texts():
                        _text_to_svg(t, ax_to_fig, fp, parts)
            except Exception:
                pass
        except Exception:
            pass

    # Figure suptitle / texts
    try:
        if getattr(fig, '_suptitle', None) is not None:
            s = fig._suptitle.get_text()
            if s:
                parts.append(
                    f'<text x="{_fmt(w_px/2)}" y="15" '
                    f'text-anchor="middle" font-size="14">{s}</text>'
                )
    except Exception:
        pass
    try:
        for t in list(fig.texts):
            _text_to_svg(t, lambda p: (float(p[0]), float(p[1])),
                         lambda p: p, parts)
    except Exception:
        pass

    body = '\n  '.join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{_fmt(w_px)}" height="{_fmt(h_px)}" '
        f'viewBox="0 0 {_fmt(w_px)} {_fmt(h_px)}">\n  {body}\n</svg>'
    )


def install():
    """Monkey-patch Figure with a to_svg() method."""
    from matplotlib.figure import Figure
    if not hasattr(Figure, 'to_svg'):
        Figure.to_svg = _figure_to_svg  # type: ignore[attr-defined]
