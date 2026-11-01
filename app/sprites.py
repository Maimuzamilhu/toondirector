"""Turns DeepSeek's SVG code into Cairo image surfaces (rendered once, reused every frame)."""
import io
import re

import cairo
import resvg_py


def _viewbox(svg):
    m = re.search(r'viewBox="([^"]+)"', svg)
    vals = [float(v) for v in re.split(r"[\s,]+", m.group(1).strip())] if m else [0, 0, 100, 100]
    return vals if len(vals) == 4 else [0, 0, 100, 100]


def render_svg(svg, zoom):
    """SVG -> ARGB Cairo surface at `zoom` pixels per SVG unit."""
    vx, vy, vw, vh = _viewbox(svg)
    sized = re.sub(r"<svg\b", f'<svg width="{vw:g}" height="{vh:g}"', svg, count=1)
    png = resvg_py.svg_to_bytes(svg_string=sized, zoom=zoom)
    return cairo.ImageSurface.create_from_png(io.BytesIO(png))


class Sprite:
    """A rendered part plus where its viewBox origin sits, in SVG units."""

    def __init__(self, svg, zoom):
        self.vx, self.vy, self.vw, self.vh = _viewbox(svg)
        self.zoom = zoom
        self.surface = render_svg(svg, zoom)

    def paint(self, ctx, mirror=False, alpha=1.0):
        """Paints in the current user space, where 1 unit = 1 SVG unit."""
        ctx.save()
        if mirror:
            ctx.scale(-1, 1)
        ctx.scale(1 / self.zoom, 1 / self.zoom)
        ctx.set_source_surface(self.surface, self.vx * self.zoom, self.vy * self.zoom)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        if alpha < 1:
            ctx.paint_with_alpha(alpha)
        else:
            ctx.paint()
        ctx.restore()


def build_rig(parts, zoom):
    out = {}
    for name, svg in parts.items():
        try:
            out[name] = Sprite(svg, zoom)
        except Exception:
            pass
    return out


def background_surface(svg, W, H):
    """Renders a background SVG to exactly W x H (stretching the viewBox to fill)."""
    vx, vy, vw, vh = _viewbox(svg)
    sized = re.sub(r"<svg\b", f'<svg width="{W}" height="{H}" preserveAspectRatio="xMidYMid slice"', svg, count=1)
    png = resvg_py.svg_to_bytes(svg_string=sized)
    return cairo.ImageSurface.create_from_png(io.BytesIO(png))
