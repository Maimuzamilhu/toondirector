"""Explainer style - the look of polished animated explainer videos.

Characters are deliberately simple (that is what makes this style clean and consistent):
round black head, two white dot eyes, a coloured capsule body, thin black stick
limbs; small accessories (cap, bun, long hair, dress, scarf, glasses) tell people apart.
The quality comes from the backgrounds and the light: soft gradients, glow, depth blur,
vignette - plus motion-graphics badges and labels on top.
"""
import math

import cairo
import cv2
import numpy as np

from .character import rgb, shade


# ----------------------------------------------------------------- character
class StickCharacter:
    """look: {"body": "#rrggbb", "accent": "#rrggbb", "accessory": one of
    none/cap/beanie/bun/long_hair/ponytail/hat/crown/turban/helmet/glasses/scarf/dress/robe/beard,
    "height": 1.0}"""

    def __init__(self, ctx, look):
        self.c = ctx
        self.look = look
        self.body = rgb(look.get("body", "#3aa6a0"))
        self.accent = rgb(look.get("accent", "#e8584a"))
        self.ink = (0.07, 0.07, 0.09)

    def _line(self, pts, w, color=None):
        c = self.c
        c.new_path()
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        c.set_line_join(cairo.LINE_JOIN_ROUND)
        if color is None:  # a faint light edge keeps black limbs readable on dark sets
            c.set_source_rgba(1, 1, 1, 0.22)
            c.set_line_width(w * 1.7)
            c.stroke_preserve()
        c.set_source_rgb(*(color or self.ink))
        c.set_line_width(w)
        c.stroke()

    @staticmethod
    def _dir(deg):
        a = math.radians(deg)
        return math.sin(a), math.cos(a)

    def draw(self, s):
        """s: x, y = feet centre on the ground (px); size = full height (px); facing +-1;
        lean, head (deg); arm_l/arm_r/leg_l/leg_r = (joint, child) deg (0 = down,
        + = towards image right); blink 0..1; look_x -1..1; talk 0..1; smile -1..1;
        brow_down, brow_up 0..1; ground = floor y for the shadow when jumping."""
        c = self.c
        H = s["size"] * self.look.get("height", 1.0)
        u = H / 100.0                    # 100 units = full height
        x, y = s["x"], s["y"]
        f = s.get("facing", 1)
        acc = self.look.get("accessory", "none")
        limb = max(2.3 * u, 1.5)
        leg_len = 34 * u
        lean = s.get("lean", 0)
        ux, uy = self._dir(lean + 180)
        body_len = 30 * u
        bw = 14 * u  # body about as wide as the head, like the reference style

        # the lowest foot always stands on the floor, whatever the legs are doing
        legs = []
        for side, sign in (("l", -1), ("r", 1)):
            h_ang, k_ang = s.get(f"leg_{side}", (0, 0))
            dx, dy = self._dir(h_ang)
            kn = (sign * 2.5 * u + dx * leg_len * 0.52, dy * leg_len * 0.52)
            dx, dy = self._dir(h_ang + k_ang)
            legs.append((sign, kn, (kn[0] + dx * leg_len * 0.5, kn[1] + dy * leg_len * 0.5)))
        hip = (x, y - max(l[2][1] for l in legs) - abs(math.sin(math.radians(lean))) * bw * 0.45)
        neck = (hip[0] + ux * body_len, hip[1] + uy * body_len)

        # soft contact shadow
        c.save()
        c.translate(x, s.get("ground", y) + u)
        c.scale(1, 0.16)
        rs = (17 + abs(math.sin(math.radians(lean))) * 20) * u
        shadow = cairo.RadialGradient(0, 0, 0, 0, 0, rs)
        shadow.add_color_stop_rgba(0, 0, 0, 0, 0.42)
        shadow.add_color_stop_rgba(1, 0, 0, 0, 0)
        c.set_source(shadow)
        c.arc(0, 0, rs, 0, 2 * math.pi)
        c.fill()
        c.restore()

        if s.get("seat"):  # a simple stool, so nobody sits on thin air
            top = hip[1] + 1.5 * u
            c.rectangle(hip[0] - 9 * u, top, 18 * u, 2.6 * u)
            c.set_source_rgb(*shade(self.accent, 0.7))
            c.fill()
            for k in (-1, 1):
                self._line([(hip[0] + k * 7 * u, top + 2 * u), (hip[0] + k * 8 * u, y)], 1.5 * u,
                           shade(self.accent, 0.5))

        # legs
        for sign, kn, ft in legs:
            hp = (hip[0] + sign * 2.5 * u, hip[1])
            knee = (hip[0] + kn[0], hip[1] + kn[1])
            foot = (hip[0] + ft[0], hip[1] + ft[1])
            self._line([hp, knee, foot], limb)
            self._line([foot, (foot[0] + f * 3.2 * u, foot[1])], limb * 1.4)

        # body: a soft capsule in the shirt colour (or a dress / long robe)
        c.save()
        c.translate(*hip)
        c.rotate(math.radians(-lean))
        c.new_path()
        if acc in ("dress", "robe"):
            hem = body_len * (0.35 if acc == "dress" else 0.85)
            flare = 1.5 if acc == "dress" else 1.05
            c.move_to(-bw * 0.45, -body_len)
            c.line_to(bw * 0.45, -body_len)
            c.curve_to(bw * 0.8, -body_len * 0.3, bw * flare * 0.9, 0, bw * flare, hem)
            c.line_to(-bw * flare, hem)
            c.curve_to(-bw * flare * 0.9, 0, -bw * 0.8, -body_len * 0.3, -bw * 0.45, -body_len)
            c.close_path()
        else:
            r = bw / 2
            c.arc(0, -body_len + r, r, math.pi, 0)
            c.arc(0, -r * 0.6, r, 0, math.pi)
            c.close_path()
        grad = cairo.LinearGradient(-bw / 2, -body_len, bw / 2, 0)  # soft side light
        grad.add_color_stop_rgb(0, *shade(self.body, 1.25))
        grad.add_color_stop_rgb(0.55, *self.body)
        grad.add_color_stop_rgb(1, *shade(self.body, 0.68))
        c.set_source(grad)
        c.fill()
        c.rectangle(-bw * 0.52, -body_len * 0.3, bw * 1.04, 2.4 * u)  # belt
        c.set_source_rgb(*self.accent)
        c.fill()
        if acc == "scarf":
            c.rectangle(-bw * 0.55, -body_len + 1.5 * u, bw * 1.1, 3.2 * u)
            c.fill()
        c.restore()

        # head
        hr = 7.5 * u
        tilt = math.radians(-(lean + s.get("head", 0)))
        hx = neck[0] + ux * hr * 1.15
        hy = neck[1] + uy * hr * 1.15
        c.save()
        c.translate(hx, hy)
        c.rotate(tilt)
        if acc in ("long_hair", "ponytail"):
            c.new_path()
            if acc == "long_hair":
                c.move_to(-hr * 1.05, 0)
                c.curve_to(-hr * 1.2, hr * 1.6, hr * 1.2, hr * 1.6, hr * 1.05, 0)
                c.close_path()
            else:
                c.arc(-f * hr * 0.95, -hr * 0.1, hr * 0.45, 0, 2 * math.pi)
            c.set_source_rgb(*self.ink)
            c.fill()
        c.arc(0, 0, hr + max(u * 0.5, 0.6), 0, 2 * math.pi)  # light edge
        c.set_source_rgba(1, 1, 1, 0.2)
        c.fill()
        skull = cairo.RadialGradient(-hr * 0.35, -hr * 0.45, hr * 0.1, 0, 0, hr)
        skull.add_color_stop_rgb(0, 0.24, 0.24, 0.28)
        skull.add_color_stop_rgb(1, *self.ink)
        c.arc(0, 0, hr, 0, 2 * math.pi)
        c.set_source(skull)
        c.fill()
        if acc == "beard":
            c.new_path()
            c.arc(0, hr * 0.36, hr * 0.86, 0.38, math.pi - 0.38)
            c.close_path()
            c.set_source_rgb(0.62, 0.57, 0.52)
            c.fill()
        # eyes: two white dots, looking where the character looks
        look = s.get("look_x", 0.3 * f)
        ex = look * hr * 0.35
        open_ = 1 - s.get("blink", 0)
        eye_y = -hr * 0.08 - s.get("brow_up", 0) * hr * 0.04
        for k in (-1, 1):
            c.save()
            c.translate(ex + k * hr * 0.34, eye_y)
            c.scale(1, max(open_, 0.12))
            c.arc(0, 0, hr * 0.2, 0, 2 * math.pi)
            c.restore()
            c.set_source_rgb(1, 1, 1)
            c.fill()
        down, up = s.get("brow_down", 0), s.get("brow_up", 0)
        if down > 0.15 or up > 0.15:  # brows carry the mood
            for k in (-1, 1):
                by = eye_y - hr * (0.36 + 0.14 * up)
                self._line([(ex + k * hr * 0.54, by - hr * 0.16 * down),
                            (ex + k * hr * 0.16, by + hr * 0.14 * down)], max(hr * 0.1, 1), (1, 1, 1))
        talk, smile = s.get("talk", 0), s.get("smile", 0)
        if talk > 0.05:  # a small white mouth
            c.save()
            c.translate(ex, hr * 0.42)
            c.scale(1, max(talk, 0.25))
            c.arc(0, 0, hr * 0.2, 0, math.pi)
            c.restore()
            c.set_source_rgb(1, 1, 1)
            c.fill()
        elif abs(smile) > 0.25:  # a thin smile or frown
            c.new_path()
            if smile > 0:
                c.arc(ex, hr * 0.32, hr * 0.24, 0.2, math.pi - 0.2)
            else:
                c.arc(ex, hr * 0.68, hr * 0.24, math.pi + 0.2, -0.2)
            c.set_source_rgb(1, 1, 1)
            c.set_line_width(max(hr * 0.09, 1))
            c.stroke()
        if acc in ("cap", "beanie", "hat", "crown", "bun", "turban", "helmet"):
            c.set_source_rgb(*self.accent)
            c.new_path()
            if acc == "cap":
                c.arc(0, -hr * 0.15, hr * 1.02, math.pi, 2 * math.pi)
                c.fill()
                c.rectangle(0 if f > 0 else -hr * 1.6, -hr * 0.3, hr * 1.6, hr * 0.28)
            elif acc == "beanie":
                c.arc(0, -hr * 0.1, hr * 1.05, math.pi, 2 * math.pi)
            elif acc == "hat":
                c.rectangle(-hr * 1.5, -hr * 0.95, hr * 3, hr * 0.25)
                c.fill()
                c.rectangle(-hr * 0.85, -hr * 1.9, hr * 1.7, hr * 1.0)
            elif acc == "crown":
                c.move_to(-hr * 0.7, -hr * 0.8)
                for k2 in range(7):
                    c.line_to(-hr * 0.7 + k2 * hr * 0.233, -hr * (1.55 if k2 % 2 == 0 else 1.1))
                c.line_to(hr * 0.7, -hr * 0.8)
                c.close_path()
                c.set_source_rgb(0.95, 0.75, 0.2)
            elif acc == "bun":
                c.arc(0, -hr * 1.05, hr * 0.42, 0, 2 * math.pi)
                c.set_source_rgb(*self.ink)
            elif acc == "turban":
                c.save()
                c.translate(0, -hr * 0.92)
                c.scale(1.1, 0.6)
                c.arc(0, 0, hr, 0, 2 * math.pi)
                c.restore()
                c.fill()
                self._line([(-hr * 0.95, -hr * 0.78), (hr * 0.9, -hr * 1.2)], max(hr * 0.16, 1),
                           shade(self.accent, 0.72))
                c.new_path()
            elif acc == "helmet":
                c.arc(0, -hr * 0.2, hr * 1.04, math.pi, 2 * math.pi)
                c.close_path()
                c.fill()
                c.move_to(-hr * 0.16, -hr * 1.15)
                c.line_to(0, -hr * 1.8)
                c.line_to(hr * 0.16, -hr * 1.15)
                c.close_path()
            c.fill()
        if acc == "glasses":
            c.set_source_rgb(0.85, 0.85, 0.85)
            c.set_line_width(max(hr * 0.08, 1))
            for k in (-1, 1):
                c.arc(ex + k * hr * 0.33, -hr * 0.08, hr * 0.28, 0, 2 * math.pi)
                c.stroke()
        c.restore()

        # arms (in front): a sleeve in the clothes colour, then the forearm and hand
        for side, sign in (("l", -1), ("r", 1)):
            a_sh, a_el = s.get(f"arm_{side}", (sign * 8, 0))
            sh = (neck[0] + sign * 3.6 * u, neck[1] + 3 * u)
            dx, dy = self._dir(a_sh)
            el = (sh[0] + dx * 13 * u, sh[1] + dy * 13 * u)
            dx, dy = self._dir(a_sh + a_el)
            wr = (el[0] + dx * 12 * u, el[1] + dy * 12 * u)
            self._line([el, wr], limb)
            self._line([sh, el], limb * 1.9, shade(self.body, 0.86))
            c.arc(wr[0], wr[1], limb * 0.95, 0, 2 * math.pi)
            c.set_source_rgb(*self.ink)
            c.fill()


# ----------------------------------------------------------------- riding and pack animals
def is_beast(name):
    n = " " + str(name).lower() + " "
    return any(w in n for w in ("camel", "caravan", "horse", "cavalry", "donkey", "mule", "ox "))


def draw_beast(ctx, s, look):
    """A side-on camel or horse, standing on the floor, legs following the walk cycle."""
    c = ctx
    name = str(look.get("name", "")).lower()
    hump = "camel" in name or "caravan" in name
    u = s["size"] / 100.0
    x, y, f = s["x"], s["y"], s.get("facing", 1)
    body, dark = rgb(look.get("body", "#c9a26b")), shade(rgb(look.get("body", "#c9a26b")), 0.72)
    c.save()
    c.translate(x, s.get("ground", y) + u)
    c.scale(1, 0.14)
    rs = 40 * u
    shadow = cairo.RadialGradient(0, 0, 0, 0, 0, rs)
    shadow.add_color_stop_rgba(0, 0, 0, 0, 0.4)
    shadow.add_color_stop_rgba(1, 0, 0, 0, 0)
    c.set_source(shadow)
    c.arc(0, 0, rs, 0, 2 * math.pi)
    c.fill()
    c.restore()
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    swing = s.get("leg_l", (0, 0))[0] * 0.5
    c.set_line_width(4.2 * u)
    for k, (hx, sw) in enumerate(((-20, swing), (-13, -swing), (15, -swing), (22, swing))):
        c.set_source_rgb(*(dark if k % 2 else body))
        c.move_to(x + f * hx * u, y - 40 * u)
        a = math.radians(sw)
        c.line_to(x + f * (hx * u + math.sin(a) * 40 * u), y - (40 - 40 * math.cos(a)) * u)
        c.stroke()
    c.set_source_rgb(*dark)  # tail
    c.set_line_width(1.8 * u)
    c.move_to(x - f * 29 * u, y - 50 * u)
    c.curve_to(x - f * 36 * u, y - 46 * u, x - f * 36 * u, y - 36 * u, x - f * 34 * u, y - 30 * u)
    c.stroke()
    c.set_source_rgb(*body)  # neck
    c.set_line_width(8.5 * u)
    c.move_to(x + f * 22 * u, y - 50 * u)
    if hump:
        c.curve_to(x + f * 38 * u, y - 46 * u, x + f * 36 * u, y - 74 * u, x + f * 42 * u, y - 80 * u)
    else:
        c.line_to(x + f * 36 * u, y - 74 * u)
    c.stroke()
    for cx, cy, rx, ry in ((0, -49, 31, 13), (-3, -61, 12, 11) if hump else (0, -49, 31, 13)):
        c.save()
        c.translate(x + f * cx * u, y + cy * u)
        c.scale(rx * u, ry * u)
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()
        c.fill()
    c.rectangle(x - 13 * u, y - 58 * u, 26 * u, 15 * u)  # blanket
    c.set_source_rgb(*rgb(look.get("accent", "#b5473a")))
    c.fill()
    c.save()  # head
    c.translate(x + f * 44 * u, y - 82 * u)
    c.scale(11 * u, 6 * u)
    c.arc(0, 0, 1, 0, 2 * math.pi)
    c.restore()
    c.set_source_rgb(*body)
    c.fill()
    c.move_to(x + f * 38 * u, y - 87 * u)  # ear
    c.line_to(x + f * 36 * u, y - 94 * u)
    c.line_to(x + f * 42 * u, y - 88 * u)
    c.close_path()
    c.set_source_rgb(*dark)
    c.fill()
    c.arc(x + f * 46 * u, y - 84 * u, 1.3 * u, 0, 2 * math.pi)
    c.set_source_rgb(*StickCharacter(c, look).ink)
    c.fill()


# ----------------------------------------------------------------- living backgrounds
def ambient_kind(description):
    """Which small moving details suit a place (from words in its description)."""
    d = description.lower()
    if any(w in d for w in ("torch", "fire", "candle", "lamp", "burning", "forge")):
        return "embers"
    if any(w in d for w in ("battle", "desert", "dust", "ruin", "sand", "war")):
        return "dust"
    if any(w in d for w in ("snow", "winter")):
        return "snow"
    if any(w in d for w in ("sky", "sunset", "sunrise", "mountain", "plateau", "field", "river", "dawn", "dusk")):
        return "birds"
    return "motes"


def ambient(c, kind, t, W, H):
    """Slow particles or birds over a still set, so no shot is ever a frozen picture."""
    colors = {"embers": (1.0, 0.62, 0.22), "dust": (0.92, 0.82, 0.62), "snow": (1, 1, 1), "motes": (1, 0.96, 0.82)}
    if kind == "birds":
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        c.set_line_width(max(H * 0.0028, 1))
        c.set_source_rgba(0.1, 0.1, 0.14, 0.75)
        for k in range(5):
            px = ((t * (0.018 + k * 0.002) + k * 0.23) % 1.3 - 0.15) * W
            py = H * (0.12 + 0.045 * k) + math.sin(t * 0.5 + k) * H * 0.012
            wing = H * 0.012 * (0.55 + 0.45 * math.sin(t * 7 + k * 2))
            c.move_to(px - H * 0.014, py - wing)
            c.line_to(px, py)
            c.line_to(px + H * 0.014, py - wing)
            c.stroke()
        return
    col = colors.get(kind, colors["motes"])
    rise = {"embers": -0.05, "dust": -0.006, "snow": 0.04, "motes": -0.012}.get(kind, -0.01)
    for k in range(26):
        a, b = (k * 0.6180339) % 1, (k * 0.3819660) % 1
        px = (a + 0.012 * math.sin(t * 0.6 + k) + t * 0.004 * (1 if k % 2 else -1)) % 1
        py = (b + rise * t * (0.6 + a)) % 1
        alpha = (0.1 + 0.28 * b) * (0.6 + 0.4 * math.sin(t * 1.5 + k))
        c.arc(px * W, py * H, H * (0.0012 + 0.0022 * a), 0, 2 * math.pi)
        c.set_source_rgba(*col, max(alpha, 0.03))
        c.fill()


# ----------------------------------------------------------------- light & lens
_POST = {}


def post_process(surface, glow=0.5, vignette=0.18, grain=0.01, blur_bg=None):
    """Bloom (bright areas glow), vignette and fine film grain on a FORMAT_RGB24 surface.
    Works in place, all in 8-bit OpenCV operations (fast enough for every frame)."""
    w, h = surface.get_width(), surface.get_height()
    surface.flush()
    buf = np.ndarray((h, w, 4), np.uint8, surface.get_data())
    key = (w, h, vignette, grain)
    if key not in _POST:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        vig = np.clip((1 - vignette * np.clip(d - 0.55, 0, 1) ** 1.5) * 255, 0, 255).astype(np.uint8)
        noise = np.random.default_rng(7).normal(0, 255 * grain, (h, w)).astype(np.float32)
        pos = np.clip(noise, 0, 255).astype(np.uint8)
        neg = np.clip(-noise, 0, 255).astype(np.uint8)
        _POST[key] = (cv2.merge([vig] * 3), cv2.merge([pos] * 3), cv2.merge([neg] * 3))
    vig, pos, neg = _POST[key]
    img = np.ascontiguousarray(buf[:, :, :3])
    if glow > 0:
        small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = cv2.convertScaleAbs(gray, alpha=4.0, beta=-820.0)  # only real lights glow (>205)
        bright = cv2.multiply(small, cv2.merge([mask] * 3), scale=1 / 255)
        k = int(max(w, h) * 0.006) | 1
        bloom = cv2.resize(cv2.GaussianBlur(bright, (k, k), 0), (w, h), interpolation=cv2.INTER_LINEAR)
        bloom = cv2.convertScaleAbs(bloom, alpha=glow)
        img = cv2.bitwise_not(cv2.multiply(cv2.bitwise_not(img), cv2.bitwise_not(bloom), scale=1 / 255))
    img = cv2.multiply(img, vig, scale=1 / 255)
    img = cv2.subtract(cv2.add(img, pos), neg)
    buf[:, :, :3] = img
    surface.mark_dirty()


def depth_blur(surface, amount):
    """Soft background blur (depth of field) for a background surface."""
    if amount <= 0:
        return surface
    w, h = surface.get_width(), surface.get_height()
    surface.flush()
    buf = np.ndarray((h, w, 4), np.uint8, surface.get_data())
    k = int(max(w, h) * amount) | 1
    buf[:] = cv2.GaussianBlur(buf, (k, k), 0)
    surface.mark_dirty()
    return surface


# ----------------------------------------------------------------- motion graphics
def badge(ctx, x, y, text, size, fill="#ffffff", ink="#1d1d22", accent="#e8584a"):
    """Calendar-style badge like '+75 YEARS' / 'POPULATION 128' (x, y = top-right corner)."""
    ctx.save()
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    pw, ph = ext.width + size * 1.1, size * 1.75
    x0 = x - pw
    r = size * 0.3
    ctx.new_path()  # shadow
    ctx.rectangle(x0 + size * 0.08, y + size * 0.14, pw, ph)
    ctx.set_source_rgba(0, 0, 0, 0.25)
    ctx.fill()
    ctx.new_path()
    ctx.arc(x0 + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.arc(x0 + pw - r, y + r, r, 1.5 * math.pi, 2 * math.pi)
    ctx.arc(x0 + pw - r, y + ph - r, r, 0, 0.5 * math.pi)
    ctx.arc(x0 + r, y + ph - r, r, 0.5 * math.pi, math.pi)
    ctx.close_path()
    ctx.set_source_rgb(*rgb(fill))
    ctx.fill()
    ctx.rectangle(x0, y, pw, size * 0.35)  # coloured top strip, like a calendar
    ctx.set_source_rgb(*rgb(accent))
    ctx.fill()
    ctx.move_to(x0 + size * 0.55 - ext.x_bearing, y + ph * 0.62 - ext.y_bearing / 2 - size * 0.12)
    ctx.set_source_rgb(*rgb(ink))
    ctx.show_text(text)
    ctx.restore()


def label(ctx, cx, y, text, size):
    """Small dark pill caption, centred at cx (like 'Chatham Islands, New Zealand')."""
    ctx.save()
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    pw, ph = ext.width + size * 1.4, size * 1.7
    x0 = cx - pw / 2
    r = ph / 2
    ctx.new_path()
    ctx.arc(x0 + r, y + r, r, 0.5 * math.pi, 1.5 * math.pi)
    ctx.arc(x0 + pw - r, y + r, r, 1.5 * math.pi, 0.5 * math.pi)
    ctx.close_path()
    ctx.set_source_rgba(0.08, 0.1, 0.16, 0.82)
    ctx.fill()
    ctx.move_to(cx - ext.width / 2 - ext.x_bearing, y + ph / 2 - ext.y_bearing / 2 - size * 0.08)
    ctx.set_source_rgb(1, 1, 1)
    ctx.show_text(text)
    ctx.restore()


def title(ctx, cx, cy, text, size, color="#ffffff"):
    """Big centred title text with a soft shadow (like 'COULD 2 PEOPLE ACTUALLY...')."""
    ctx.save()
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    x = cx - ext.width / 2 - ext.x_bearing
    y = cy - ext.height / 2 - ext.y_bearing
    ctx.move_to(x + size * 0.05, y + size * 0.07)
    ctx.set_source_rgba(0, 0, 0, 0.45)
    ctx.show_text(text)
    ctx.move_to(x, y)
    ctx.set_source_rgb(*rgb(color))
    ctx.show_text(text)
    ctx.restore()



def draw_animal(ctx, kind, s, look):
    """Simple animal shapes in the same flat style: bird, seal (blob), four-legged animal."""
    c = ctx
    size = s["size"]
    x, y, f = s["x"], s["y"], s.get("facing", 1)
    body = rgb(look.get("body", "#333333"))
    ink = (0.07, 0.07, 0.09)
    c.save()
    c.translate(x, y)
    c.scale(f, 1)
    if kind == "bird":
        r = size * 0.32
        c.new_path()
        c.save()
        c.translate(0, -r * 1.1)
        c.scale(1.25, 1.0)
        c.arc(0, 0, r, 0, 2 * math.pi)
        c.restore()
        c.set_source_rgb(*body)
        c.fill()
        c.move_to(-r * 1.1, -r * 1.2)  # tail
        c.line_to(-r * 2.1, -r * 1.6)
        c.line_to(-r * 2.0, -r * 0.9)
        c.close_path()
        c.fill()
        c.move_to(r * 1.2, -r * 1.35)  # beak
        c.line_to(r * 1.75, -r * 1.2)
        c.line_to(r * 1.2, -r * 1.05)
        c.close_path()
        c.set_source_rgb(0.95, 0.7, 0.2)
        c.fill()
        c.arc(r * 0.75, -r * 1.4, r * 0.13, 0, 2 * math.pi)
        c.set_source_rgb(1, 1, 1)
        c.fill()
        c.set_source_rgb(*ink)
        c.set_line_width(max(r * 0.08, 1))
        for k in (-0.3, 0.2):
            c.move_to(r * k, -r * 0.2)
            c.line_to(r * k, 0)
            c.stroke()
    elif kind == "seal":
        L, Hh = size * 1.6, size * 0.55
        c.new_path()
        c.move_to(-L / 2, 0)
        c.curve_to(-L / 2, -Hh * 0.9, L * 0.2, -Hh * 1.1, L / 2, -Hh * 0.6)
        c.curve_to(L * 0.6, -Hh * 0.2, L * 0.4, 0, L * 0.3, 0)
        c.close_path()
        g = cairo.LinearGradient(0, -Hh, 0, 0)
        g.add_color_stop_rgb(0, *shade(body, 1.15))
        g.add_color_stop_rgb(1, *shade(body, 0.75))
        c.set_source(g)
        c.fill()
        c.arc(L * 0.38, -Hh * 0.62, Hh * 0.07, 0, 2 * math.pi)
        c.set_source_rgb(*ink)
        c.fill()
    else:  # four-legged animal (cat, dog, ...): round body, head with ears, tail, short legs
        L, Hh = size * 1.25, size * 0.5
        lw = max(size * 0.07, 1.5)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        c.set_source_rgb(*shade(body, 0.85))  # tail, curling up behind
        c.set_line_width(lw * 1.3)
        c.move_to(-L * 0.42, -Hh * 0.95)
        c.curve_to(-L * 0.75, -Hh * 1.1, -L * 0.7, -Hh * 2.0, -L * 0.45, -Hh * 2.05)
        c.stroke()
        c.set_source_rgb(*shade(body, 0.7))  # legs
        c.set_line_width(lw)
        for k in (-0.3, -0.15, 0.18, 0.32):
            c.move_to(L * k, -Hh * 0.7)
            c.line_to(L * k, -lw * 0.4)
            c.stroke()
        c.new_path()  # body
        c.save()
        c.translate(0, -Hh * 1.0)
        c.scale(L / 2, Hh / 2.1)
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()
        g = cairo.LinearGradient(0, -Hh * 1.5, 0, -Hh * 0.5)
        g.add_color_stop_rgb(0, *shade(body, 1.12))
        g.add_color_stop_rgb(1, *shade(body, 0.85))
        c.set_source(g)
        c.fill()
        hx, hy, hr = L * 0.48, -Hh * 1.55, Hh * 0.55  # head
        for sgn in (-1, 1):  # ears
            c.move_to(hx + sgn * hr * 0.75, hy - hr * 0.35)
            c.line_to(hx + sgn * hr * 0.55, hy - hr * 1.35)
            c.line_to(hx + sgn * hr * 0.05, hy - hr * 0.8)
            c.close_path()
            c.set_source_rgb(*shade(body, 0.95))
            c.fill()
        c.arc(hx, hy, hr, 0, 2 * math.pi)
        c.set_source_rgb(*shade(body, 1.05))
        c.fill()
        for sgn in (-1, 1):  # eyes and a small nose
            c.arc(hx + sgn * hr * 0.35 + hr * 0.15, hy - hr * 0.1, hr * 0.13, 0, 2 * math.pi)
            c.set_source_rgb(*ink)
            c.fill()
        c.arc(hx + hr * 0.15, hy + hr * 0.25, hr * 0.09, 0, 2 * math.pi)
        c.set_source_rgb(0.85, 0.4, 0.45)
        c.fill()
    c.restore()


def draw_crowd(ctx, s, look):
    """A small crowd: a row of simple figures in different colours around (x, y)."""
    palette = ["#3aa6a0", "#e8584a", "#e0a42b", "#7b4bc4", "#4a8fd9", "#d96a9a", "#5aa05a", "#c98a4a"]
    size = s["size"]
    n = 7
    spread = size * 0.38
    accs = ["none", "cap", "bun", "none", "long_hair", "beanie", "none"]
    for k in range(n):
        dx = (k - (n - 1) / 2) * spread
        sz = size * (0.92 + 0.12 * ((k * 37) % 5) / 4)
        figure = StickCharacter(ctx, {"body": palette[(k * 3) % len(palette)], "accent": palette[(k * 5 + 2) % len(palette)],
                                      "accessory": accs[k]})
        st = dict(s, x=s["x"] + dx, size=sz, facing=1 if k % 2 else -1, look_x=0)
        figure.draw(st)
