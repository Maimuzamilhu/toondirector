"""Draws the 2D cartoon character with Cairo. One call = one video frame.

Head parts are drawn in 'head units': head radius = 1, origin = head centre.
"""
import math

import cairo


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def shade(c, f):
    return tuple(min(max(x * f, 0), 1) for x in c)


class Painter:
    """Draws one video frame: the shot's background, then every puppet."""

    def __init__(self, W, H, puppets, backgrounds=None, bg_style=None, foregrounds=None, full_body=False):
        self.W, self.H = W, H
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.puppets = puppets            # list of (style, rig) per person
        self.backgrounds = backgrounds or {}  # group id -> cairo surface (W x H)
        self.foregrounds = foregrounds or {}  # group id -> table / props in front of people
        st = bg_style or {"bg_top": "#bfe3ff", "bg_bottom": "#f4f9ff"}
        self.grad = cairo.LinearGradient(0, 0, 0, H)
        self.grad.add_color_stop_rgb(0, *rgb(st["bg_top"]))
        self.grad.add_color_stop_rgb(1, *rgb(st["bg_bottom"]))
        cls = FullPuppet if full_body else Puppet
        self._puppets = [cls(self.ctx, W, H, style, rig) for style, rig in puppets]

    def draw(self, states, bg_group=None, subtitle=None, zoom=1.0, effects=()):
        """states: one state dict per person (None = person not in this frame).
        zoom > 1 pushes the camera in slightly (around the frame centre)."""
        c = self.ctx
        c.save()
        if zoom != 1.0:
            c.translate(self.W / 2, self.H / 2)
            c.scale(zoom, zoom)
            c.translate(-self.W / 2, -self.H / 2)
        bg = self.backgrounds.get(bg_group)
        if bg is not None:
            c.set_source_surface(bg, 0, 0)
            c.paint()
        else:
            c.set_source(self.grad)
            c.paint()
        order = sorted((i for i, s in enumerate(states) if s is not None),
                       key=lambda i: states[i].get("r", states[i].get("k", 0) * 100))
        for i in order:  # smaller (further away) people first
            s = states[i]
            if s.get("alpha", 1) < 0.02:
                continue
            pup = self._puppets[i]
            fn = pup.draw_full if isinstance(pup, FullPuppet) else pup.draw
            if s.get("alpha", 1) < 0.98:
                c.push_group()
                fn(s)
                c.pop_group_to_source()
                c.paint_with_alpha(s["alpha"])
            else:
                fn(s)
        for x1, y1, x2, y2, color, width in effects:  # lasso ropes, laser beams, ...
            c.set_line_cap(cairo.LINE_CAP_ROUND)
            c.move_to(x1, y1)
            c.line_to(x2, y2)
            c.set_source_rgba(*rgb(color), 0.35)
            c.set_line_width(width * 2.2)
            c.stroke_preserve()
            c.set_source_rgb(*rgb(color))
            c.set_line_width(width)
            c.stroke()
        fg = self.foregrounds.get(bg_group)
        if fg is not None:  # the table and props go in front of the characters
            c.set_source_surface(fg, 0, 0)
            c.paint()
        c.restore()
        if subtitle:
            self._subtitle(subtitle)
        self.surface.flush()
        return self.surface.get_data()

    def png(self, path):
        self.surface.write_to_png(str(path))

    def _subtitle(self, text):
        c, W, H = self.ctx, self.W, self.H
        size = max(H * 0.045, 14)
        c.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        c.set_font_size(size)
        words, lines, cur = text.split(), [], ""
        for wd in words:
            test = (cur + " " + wd).strip()
            if c.text_extents(test).width > W * 0.86 and cur:
                lines.append(cur)
                cur = wd
            else:
                cur = test
        if cur:
            lines.append(cur)
        lines = lines[-2:]
        y = H - size * 0.8 - (len(lines) - 1) * size * 1.25
        for line in lines:
            ext = c.text_extents(line)
            x = (W - ext.width) / 2 - ext.x_bearing
            c.new_path()
            c.move_to(x, y)
            c.text_path(line)
            c.set_source_rgb(0, 0, 0)
            c.set_line_width(size * 0.18)
            c.stroke_preserve()
            c.set_source_rgb(1, 1, 1)
            c.fill()
            y += size * 1.25


class Puppet:
    """One character. Uses DeepSeek's SVG parts (rig) where present, built-in drawing otherwise."""

    def __init__(self, ctx, W, H, style, rig=None):
        self.ctx, self.W, self.H = ctx, W, H
        self.rig = rig or {}
        st = style
        self.skin = rgb(st["skin"])
        self.skin_dark = shade(self.skin, 0.82)
        self.hair = rgb(st["hair_color"])
        self.eye = rgb(st["eye_color"])
        self.shirt = rgb(st["shirt_color"])
        self.beard_c = rgb(st["beard_color"])
        self.line = rgb(st["outline"])
        self.brow = rgb(st["brow_color"]) if st.get("brow_color") else None
        # dark clothes need a lighter edge or arms disappear into the body
        lum = 0.3 * self.shirt[0] + 0.59 * self.shirt[1] + 0.11 * self.shirt[2]
        self.cloth_line = tuple(min(x * 1.8 + 0.22, 1) for x in self.shirt) if lum < 0.3 else self.line
        self.style = st

    # ---------------- helpers ----------------
    def _fs(self, color, lw, outline=None):
        c = self.ctx
        c.set_source_rgb(*color)
        c.fill_preserve()
        c.set_source_rgb(*(outline or self.line))
        c.set_line_width(lw)
        c.stroke()

    def _ellipse(self, x, y, rx, ry):
        c = self.ctx
        c.save()
        c.translate(x, y)
        c.scale(max(rx, 1e-4), max(ry, 1e-4))
        c.new_sub_path()
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()

    def _limb(self, pts, width, color, lw):
        c = self.ctx
        c.new_path()
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.set_source_rgb(*self.cloth_line)
        c.set_line_width(width + 2 * lw)
        c.stroke_preserve()
        c.set_source_rgb(*color)
        c.set_line_width(width)
        c.stroke()

    def _sprite_unit(self, name, mirror=False):
        """Paints a rig part in head space (1 unit = head radius; parts use radius 100)."""
        sp = self.rig.get(name)
        if sp is None:
            return False
        c = self.ctx
        c.save()
        c.scale(0.01, 0.01)
        sp.paint(c, mirror)
        c.restore()
        return True

    def _bone(self, name, a, b, rest_len, k, mirror):
        """Places a limb part so its (0,0) sits on a and its +y axis points at b."""
        sp = self.rig.get(name)
        if sp is None:
            return False
        c = self.ctx
        vx, vy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(vx, vy)
        stretch = min(max(length / max(rest_len * k, 1e-3), 0.55), 1.35)
        c.save()
        c.translate(*a)
        c.rotate(math.atan2(-vx, vy))
        c.scale(k, k * stretch)
        sp.paint(c, mirror)
        c.restore()
        return True

    # ---------------- person ----------------
    def draw(self, s):
        c, H = self.ctx, self.H
        r = s["r"]
        lw = max(r * 0.035, 1.2)
        k = r / 100.0  # pixels per rig unit

        if "hair_back" in self.rig:
            self._head_space(s, lambda s_, lw_: self._sprite_unit("hair_back"))
        elif self.style["hair_style"] in ("long", "hijab"):
            self._head_space(s, self._hair_back)

        lx, ly, rx, ry = s["lsh_x"], s["lsh_y"], s["rsh_x"], s["rsh_y"]
        if lx > rx:
            lx, ly, rx, ry = rx, ry, lx, ly
        hx, hy = s["hx"], s["hy"]

        if "torso" in self.rig:
            # torso scale follows shoulder width, kept close to the head's scale
            kt = min(max(math.hypot(rx - lx, ry - ly) / 230.0, k * 0.8), k * 1.4)
            mx, my = (lx + rx) / 2, (ly + ry) / 2 - 25 * kt
            nw = 26 * kt
            c.new_path()  # neck, under the collar
            c.move_to(hx - nw, hy + r * 0.5)
            c.line_to(mx - nw, my + 10 * kt)
            c.line_to(mx + nw, my + 10 * kt)
            c.line_to(hx + nw, hy + r * 0.5)
            c.close_path()
            self._fs(self.skin_dark, lw)
            ang = math.atan2(ry - ly, rx - lx)
            # the drawn torso ends ~4.5 head-radii down; in wide shots extend the body to
            # the bottom of the frame (drawn first, so the torso drawing covers the join)
            bottom = my + 440 * kt
            if bottom < H:
                half = 120 * kt
                c.new_path()
                c.move_to(mx - half, my + 300 * kt)
                c.line_to(mx + half, my + 300 * kt)
                c.line_to(mx + half * 1.05, H + 20)
                c.line_to(mx - half * 1.05, H + 20)
                c.close_path()
                self._fs(shade(self.shirt, 0.85), lw, self.cloth_line)
            c.save()
            c.translate(mx, my)
            c.rotate(ang)
            c.scale(kt, kt)
            self.rig["torso"].paint(c)
            c.restore()
        else:
            kt = k
            self._template_torso(s, lx, ly, rx, ry, r, lw)

        self._head_space(s, self._head)

        # arms in front, so raised hands can pass in front of the body and face
        for side in ("l", "r"):
            S = (s[f"{side}sh_x"], s[f"{side}sh_y"])
            E = (s[f"{side}el_x"], s[f"{side}el_y"])
            Wr = (s[f"{side}wr_x"], s[f"{side}wr_y"])
            mirror = side == "l"
            holds_mic = s.get(f"mic_{side}", 0) > 0.5
            if "upper_arm" in self.rig and "forearm" in self.rig:
                self._bone("upper_arm", S, E, 140, kt, mirror)
                self._bone("forearm", E, Wr, 125, kt, mirror)
            else:
                self._limb([S, E, Wr], r * 0.36, self.shirt, lw)
            if holds_mic:
                # mic over the forearm, then a closed fist gripping it
                self._mic(Wr, E, r)
                self._ellipse(Wr[0], Wr[1], r * 0.15, r * 0.13)
                self._fs(self.skin, lw)
                continue
            if "upper_arm" in self.rig and "forearm" in self.rig:
                hand_to = (Wr[0] + (Wr[0] - E[0]) * 0.3, Wr[1] + (Wr[1] - E[1]) * 0.3)
                if self._bone("hand", Wr, hand_to, 40, kt, mirror):
                    continue
            self._ellipse(Wr[0], Wr[1], r * 0.17, r * 0.17)
            self._fs(self.skin, lw)

    def _mic(self, Wr, E, r):
        """Handheld microphone: grip in the hand, head pointing up towards the mouth."""
        c = self.ctx
        lw = max(r * 0.03, 1)
        ang = math.atan2(Wr[1] - E[1], Wr[0] - E[0])
        c.save()
        c.translate(*Wr)
        c.rotate(ang + math.pi / 2)  # mic head continues the elbow->wrist line
        c.scale(r / 100, r / 100)
        c.new_path()  # grip
        c.rectangle(-9, -10, 18, 70)
        self._fs((0.12, 0.12, 0.14), 3)
        c.new_path()  # coloured flag
        c.rectangle(-16, -26, 32, 22)
        self._fs((0.18, 0.35, 0.85), 3)
        c.new_path()  # mesh ball
        c.arc(0, -42, 18, 0, 2 * math.pi)
        self._fs((0.55, 0.57, 0.6), 3)
        c.set_source_rgba(0.3, 0.3, 0.32, 0.8)
        c.set_line_width(2)
        for k in (-8, 0, 8):
            c.move_to(-16, -42 + k)
            c.line_to(16, -42 + k)
        c.stroke()
        c.restore()

    def _template_torso(self, s, lx, ly, rx, ry, r, lw):
        c, H = self.ctx, self.H
        pad = r * 0.28
        c.new_path()
        c.move_to(lx - pad, H + 50)
        c.line_to(lx - pad, ly + pad)
        c.curve_to(lx - pad, ly - pad * 0.4, lx - pad * 0.2, ly - pad * 0.7, lx + pad, ly - pad * 0.7)
        c.line_to(rx - pad, ry - pad * 0.7)
        c.curve_to(rx + pad * 0.2, ry - pad * 0.7, rx + pad, ry - pad * 0.4, rx + pad, ry + pad)
        c.line_to(rx + pad, H + 50)
        c.close_path()
        self._fs(self.shirt, lw, self.cloth_line)
        hx, hy = s["hx"], s["hy"]
        mx, my = (lx + rx) / 2, (ly + ry) / 2 - pad * 0.5
        nw = r * 0.24
        c.new_path()
        c.move_to(hx - nw, hy + r * 0.5)
        c.line_to(mx - nw, my)
        c.line_to(mx + nw, my)
        c.line_to(hx + nw, hy + r * 0.5)
        c.close_path()
        self._fs(self.skin_dark, lw)
        c.new_path()
        c.move_to(mx - nw * 1.6, my - pad * 0.1)
        c.line_to(mx, my + pad * 0.9)
        c.line_to(mx + nw * 1.6, my - pad * 0.1)
        c.set_source_rgb(*shade(self.shirt, 0.7))
        c.set_line_width(lw * 2)
        c.stroke()

    def _head_space(self, s, fn):
        c = self.ctx
        c.save()
        c.translate(s["hx"], s["hy"])
        c.rotate(s["roll"])
        c.scale(s["r"], s["r"])
        fn(s, 0.035)
        c.restore()

    # ---------------- head (unit space) ----------------
    def _hair_back(self, s, lw):
        c = self.ctx
        c.new_path()
        if self.style["hair_style"] == "hijab":
            c.move_to(-1.15, -0.2)
            c.curve_to(-1.25, -1.45, 1.25, -1.45, 1.15, -0.2)
            c.curve_to(1.2, 0.9, 1.9, 1.5, 2.0, 2.2)
            c.line_to(2.1, 6.0)
            c.line_to(-2.1, 6.0)
            c.line_to(-2.0, 2.2)
            c.curve_to(-1.9, 1.5, -1.2, 0.9, -1.15, -0.2)
        else:
            c.move_to(-1.0, -0.3)
            c.curve_to(-1.1, -1.3, 1.1, -1.3, 1.0, -0.3)
            c.curve_to(1.1, 0.5, 1.25, 1.2, 1.2, 1.7)
            c.line_to(-1.2, 1.7)
            c.curve_to(-1.25, 1.2, -1.1, 0.5, -1.0, -0.3)
        c.close_path()
        self._fs(self.hair, lw)

    def _head(self, s, lw):
        c = self.ctx
        yaw, pitch = s["yaw"], s["pitch"]
        dx, dy = yaw * 0.3, pitch * 0.22
        hw = 0.88 * (1 - 0.06 * abs(yaw))
        style = self.style
        rig = self.rig

        if "head" in rig:
            c.save()
            c.scale(1 - 0.06 * abs(yaw), 1)  # slight squash when turning
            self._sprite_unit("head")
            c.restore()
        else:
            if style["hair_style"] == "bun":
                self._ellipse(0, -1.2, 0.34, 0.3)
                self._fs(self.hair, lw)
            if style["hair_style"] != "hijab":
                for sign in (-1, 1):
                    ex = sign * hw * 0.98 - dx * 0.6
                    self._ellipse(ex, 0.08 + dy * 0.3, 0.15, 0.2)
                    self._fs(self.skin, lw)
            c.new_path()
            c.move_to(0, -1.0)
            c.curve_to(hw * 0.75, -1.0, hw, -0.6, hw, -0.05)
            c.curve_to(hw, 0.55, hw * 0.55, 1.0, 0, 1.0)
            c.curve_to(-hw * 0.55, 1.0, -hw, 0.55, -hw, -0.05)
            c.curve_to(-hw, -0.6, -hw * 0.75, -1.0, 0, -1.0)
            c.close_path()
            self._fs(self.skin, lw)

        if s["blush"] > 0.02:
            for sign in (-1, 1):
                self._ellipse(sign * 0.5 + dx, 0.3 + dy, 0.15, 0.08)
                c.set_source_rgba(1.0, 0.45, 0.5, 0.55 * s["blush"])
                c.fill()

        has_details = "face_details" in rig
        if style["beard"] and not has_details:
            self._beard(dx, dy, lw)

        self._eyes(s, dx, dy, lw)
        self._brows(s, dx, dy)

        if has_details:
            # nose/beard/glasses drawn by DeepSeek; they follow the face turn
            c.save()
            c.translate(dx, dy)
            self._sprite_unit("face_details")
            c.restore()
        else:
            nx, ny = dx * 1.25, 0.22 + dy
            c.new_path()
            c.move_to(nx - 0.02, ny - 0.1)
            c.curve_to(nx + 0.08 * (1 if yaw >= 0 else -1), ny, nx + 0.04, ny + 0.06, nx - 0.04, ny + 0.05)
            c.set_source_rgb(*shade(self.skin, 0.6))
            c.set_line_width(lw)
            c.stroke()

        self._mouth(s, dx, dy, lw)

        if style["glasses"] and not has_details:
            c.set_source_rgb(*self.line)
            c.set_line_width(lw * 1.4)
            for sign in (-1, 1):
                self._ellipse(sign * 0.33 + dx, -0.05 + dy, 0.22, 0.19)
                c.stroke()
            c.move_to(-0.11 + dx, -0.08 + dy)
            c.curve_to(-0.05 + dx, -0.14 + dy, 0.05 + dx, -0.14 + dy, 0.11 + dx, -0.08 + dy)
            c.stroke()

        if "hair_front" in rig:
            c.save()
            c.translate(dx * 0.4, 0)
            self._sprite_unit("hair_front")
            c.restore()
        else:
            self._hair_front(dx, lw, hw)

    def _eyes(self, s, dx, dy, lw):
        c = self.ctx
        for side, sign in (("l", -1), ("r", 1)):
            ex, ey = sign * 0.33 + dx, -0.05 + dy
            open_ = min(max(1 - s[f"blink_{side}"] * 1.25 + s["eye_wide"] * 0.3, 0), 1.3)
            ew, eh = 0.16, 0.2 * open_
            if open_ < 0.2:
                c.new_path()
                c.move_to(ex - ew, ey)
                c.curve_to(ex - ew * 0.4, ey + 0.06, ex + ew * 0.4, ey + 0.06, ex + ew, ey)
                c.set_source_rgb(*self.line)
                c.set_line_width(lw * 1.2)
                c.stroke()
                continue
            self._ellipse(ex, ey, ew, eh)
            self._fs((1, 1, 1), lw)
            c.save()
            self._ellipse(ex, ey, ew, eh)
            c.clip()
            px = ex + s["gaze_x"] * 0.07
            py = ey + s["gaze_y"] * 0.07
            self._ellipse(px, py, 0.1, 0.12)
            c.set_source_rgb(*self.eye)
            c.fill()
            self._ellipse(px, py, 0.055, 0.065)
            c.set_source_rgb(0.05, 0.05, 0.05)
            c.fill()
            self._ellipse(px + 0.035, py - 0.045, 0.028, 0.028)
            c.set_source_rgb(1, 1, 1)
            c.fill()
            # happy "eye smile": cheek pushes the lower lid up
            if s["eye_smile"] > 0.02:
                self._ellipse(ex, ey + eh * (1.75 - s["eye_smile"] * 0.9), ew * 1.5, eh)
                c.set_source_rgb(*self.skin)
                c.fill()
            c.restore()
            self._ellipse(ex, ey, ew, eh)
            c.set_source_rgb(*self.line)
            c.set_line_width(lw)
            c.stroke()

    def _brows(self, s, dx, dy):
        c = self.ctx
        if self.brow:
            color = self.brow
        else:
            color = shade(self.hair, 0.8) if self.style["hair_style"] not in ("bald", "hijab") else shade(self.skin, 0.45)
        base = -0.33 - s["brow_up"] * 0.13 + s["brow_down"] * 0.07 + dy
        tilt = s["brow_tilt"] * 0.07 - s["brow_down"] * 0.05
        c.set_source_rgb(*color)
        c.set_line_width(0.065)
        for sign in (-1, 1):
            cx = sign * 0.33 + dx
            inner_x, outer_x = cx - sign * 0.14, cx + sign * 0.15
            c.new_path()
            c.move_to(outer_x, base + 0.02)
            c.curve_to(cx, base - 0.05, cx, base - 0.04, inner_x, base - tilt)
            c.stroke()

    def _mouth(self, s, dx, dy, lw):
        c = self.ctx
        mx, my = dx * 1.1, 0.5 + dy * 0.9
        o, wd, rd, sm = s["open"], s["width"], s["round"], s["smile"]
        half = 0.19 * (1 + 0.35 * wd) * (1 - 0.4 * rd)
        h = 0.42 * o
        cy = my - sm * 0.07
        if self.style["beard"]:  # clear the lip area so the mouth reads on a dark beard
            self._ellipse(mx, my + h * 0.4, half * 1.35, 0.1 + h * 0.6)
            c.set_source_rgb(*self.skin)
            c.fill()
        if o < 0.06:
            c.new_path()
            c.move_to(mx - half, cy)
            c.curve_to(mx - half * 0.4, my + sm * 0.09, mx + half * 0.4, my + sm * 0.09, mx + half, cy)
            c.set_source_rgb(*shade(self.skin, 0.35))
            c.set_line_width(lw * 1.3)
            c.stroke()
            return
        k = 0.55 + 0.35 * rd
        c.new_path()
        c.move_to(mx - half, cy)
        c.curve_to(mx - half * k, my - h * 0.35, mx + half * k, my - h * 0.35, mx + half, cy)
        c.curve_to(mx + half * k, my + h + sm * 0.05, mx - half * k, my + h + sm * 0.05, mx - half, cy)
        c.close_path()
        c.set_source_rgb(0.32, 0.07, 0.1)
        c.fill_preserve()
        c.save()
        c.clip_preserve()
        if o > 0.1:
            c.rectangle(mx - half, my - h * 0.35 - 0.1, half * 2, 0.1 + min(h * 0.25, 0.08))
            c.set_source_rgb(1, 1, 1)
            c.fill()
        if o > 0.25:
            self._ellipse(mx, my + h * 0.85, half * 0.6, h * 0.35)
            c.set_source_rgb(0.9, 0.4, 0.45)
            c.fill()
        c.restore()
        c.new_path()
        c.move_to(mx - half, cy)
        c.curve_to(mx - half * k, my - h * 0.35, mx + half * k, my - h * 0.35, mx + half, cy)
        c.curve_to(mx + half * k, my + h + sm * 0.05, mx - half * k, my + h + sm * 0.05, mx - half, cy)
        c.close_path()
        c.set_source_rgb(*shade(self.skin, 0.35))
        c.set_line_width(lw)
        c.stroke()

    def _beard(self, dx, dy, lw):
        c = self.ctx
        c.new_path()
        c.move_to(-0.86, 0.0)
        c.curve_to(-0.9, 0.75, -0.5, 1.12, 0, 1.12)
        c.curve_to(0.5, 1.12, 0.9, 0.75, 0.86, 0.0)
        c.curve_to(0.7, 0.32, 0.45, 0.36, 0.15 + dx, 0.36 + dy)
        c.curve_to(0.05 + dx, 0.33 + dy, -0.05 + dx, 0.33 + dy, -0.15 + dx, 0.36 + dy)
        c.curve_to(-0.45, 0.36, -0.7, 0.32, -0.86, 0.0)
        c.close_path()
        self._fs(self.beard_c, lw)

    def _hair_front(self, dx, lw, hw):
        c = self.ctx
        st = self.style["hair_style"]
        f = dx * 0.5
        if st == "bald":
            return
        if st == "hijab":
            # scarf frame around the face (outer shape minus face opening)
            c.new_path()
            c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            c.move_to(-1.15, 0.2)
            c.curve_to(-1.3, -1.45, 1.3, -1.45, 1.15, 0.2)
            c.curve_to(1.1, 0.9, 0.6, 1.25, 0, 1.25)
            c.curve_to(-0.6, 1.25, -1.1, 0.9, -1.15, 0.2)
            c.close_path()
            c.new_sub_path()
            c.move_to(0 + f, -0.78)
            c.curve_to(0.62 + f, -0.78, 0.8, -0.3, 0.8, 0.1)
            c.curve_to(0.8, 0.6, 0.45, 0.98, 0, 0.98)
            c.curve_to(-0.45, 0.98, -0.8, 0.6, -0.8, 0.1)
            c.curve_to(-0.8, -0.3, -0.62 + f, -0.78, 0 + f, -0.78)
            c.close_path()
            self._fs(self.hair, lw)
            c.set_fill_rule(cairo.FILL_RULE_WINDING)
            return
        if st == "curly":
            for i in range(13):
                a = math.pi * (1.06 + i * 0.88 / 12)
                self._ellipse(math.cos(a) * 0.95 + f * (1 - abs(math.cos(a))), math.sin(a) * 0.95 - 0.05, 0.26, 0.26)
                self._fs(self.hair, lw)
            return
        c.new_path()
        c.move_to(-hw - 0.06, 0.05)
        c.curve_to(-1.05, -0.95, -0.5, -1.2, 0, -1.17)
        c.curve_to(0.5, -1.2, 1.05, -0.95, hw + 0.06, 0.05)
        if st == "long":
            c.curve_to(hw * 0.95, -0.35, 0.7, -0.5, 0.45 + f, -0.5)
            c.curve_to(0.2 + f, -0.6, -0.15 + f, -0.48, -0.3 + f, -0.58)
            c.curve_to(-0.6, -0.55, -hw * 0.95, -0.35, -hw - 0.06, 0.05)
        else:
            c.curve_to(hw * 0.92, -0.42, 0.72, -0.52, 0.48 + f, -0.5)
            c.curve_to(0.3 + f, -0.66, 0.05 + f, -0.5, -0.12 + f, -0.6)
            c.curve_to(-0.35 + f, -0.68, -0.6, -0.5, -0.75, -0.55)
            c.curve_to(-0.85, -0.45, -hw * 0.98, -0.3, -hw - 0.06, 0.05)
        c.close_path()
        self._fs(self.hair, lw)
        if st == "spiky":
            for i in range(7):
                x = -0.75 + i * 0.25
                base = -1.0 + abs(x) * 0.35
                c.new_path()
                c.move_to(x - 0.14, base + 0.05)
                c.line_to(x + 0.05 + f * 0.3, base - 0.38 + abs(x) * 0.15)
                c.line_to(x + 0.14, base + 0.05)
                c.close_path()
                self._fs(self.hair, lw)


class FullPuppet(Puppet):
    """Full-body character (animator mode), posed by joint angles instead of tracked points.

    State keys: x, y (hip centre, px), k (px per rig unit; head radius = 100 units),
    lean, head (deg), arm_l/arm_r/leg_l/leg_r = (joint, child joint) deg, facing (+-1),
    plus the face keys the head drawing uses (open, smile, blink_l, ...)."""

    TORSO = 430     # neckline (torso 0,0) to hip line, rig units
    NECK = 165      # head centre above the neckline
    SHOULDER = (115, 25)
    HIP = 55
    UPPER, FORE = 140, 125
    THIGH, SHIN = 200, 190

    @staticmethod
    def _dir(deg):
        a = math.radians(deg)
        return math.sin(a), math.cos(a)  # 0 deg = straight down, + = towards image right

    def _seg(self, name, a, ang, length, k, mirror, width_frac=0.36):
        """Draws a limb part from point a along angle ang; returns the end point."""
        dx, dy = self._dir(ang)
        b = (a[0] + dx * length * k, a[1] + dy * length * k)
        if not self._bone(name, a, b, length, k, mirror):
            self._limb([a, b], length * k * width_frac * 0.6, self.shirt, max(k * 3.5, 1.2))
        return b

    def draw_full(self, s):
        c = self.ctx
        k = s["k"]
        hx, hy = s["x"], s["y"]
        lean = s["lean"]
        r = 100 * k
        lw = max(r * 0.035, 1.2)
        ux, uy = self._dir(lean + 180)  # "up" along the torso
        neck = (hx + ux * self.TORSO * k, hy + uy * self.TORSO * k)
        px, py = math.cos(math.radians(-lean)), math.sin(math.radians(-lean))  # sideways along the hips

        def side(pt, off_x, off_y=0.0):
            return (pt[0] + px * off_x * k - ux * off_y * k * 0, pt[1] + py * off_x * k + off_y * k)

        # legs first (behind the torso)
        for sd, sign in (("l", -1), ("r", 1)):
            hip = (hx + px * sign * self.HIP * k, hy + py * sign * self.HIP * k)
            h_ang, k_ang = s[f"leg_{sd}"]
            knee = self._seg("thigh", hip, h_ang, self.THIGH, k, sd == "l", 0.42)
            ankle = self._seg("shin", knee, h_ang + k_ang, self.SHIN, k, sd == "l", 0.36)
            foot = self.rig.get("foot")
            if foot is not None:
                c.save()
                c.translate(*ankle)
                c.scale(k * s["facing"], k)
                foot.paint(c)
                c.restore()
            else:
                self._ellipse(ankle[0] + s["facing"] * 30 * k, ankle[1] + 25 * k, 55 * k, 22 * k)
                self._fs(shade(self.shirt, 0.4), lw)

        # torso, hanging from the neckline
        if "torso" in self.rig:
            c.save()
            c.translate(*neck)
            c.rotate(math.radians(-lean))
            c.scale(k, k)
            self.rig["torso"].paint(c)
            c.restore()
        else:
            c.new_path()
            for i, (ox, oy) in enumerate(((-115, 0), (115, 0), (70, self.TORSO + 20), (-70, self.TORSO + 20))):
                x = neck[0] + px * ox * k - ux * oy * k
                y = neck[1] + py * ox * k - uy * oy * k
                (c.move_to if i == 0 else c.line_to)(x, y)
            c.close_path()
            self._fs(self.shirt, lw, self.cloth_line)

        # head (reuses the normal head drawing: face, eyes, mouth, hair)
        hs = dict(s)
        big = s.get("head_scale", 1.0)  # caricature heads
        hs["r"] = r * big
        lift = (self.NECK - 100 + 100 * big) * k  # neck length stays, a bigger head sits higher
        hs["hx"] = neck[0] + ux * lift
        hs["hy"] = neck[1] + uy * lift
        hs["roll"] = math.radians(-(lean + s["head"]))
        hs["yaw"] = 0.35 * s["facing"]
        self._head_space(hs, self._head)

        # arms in front
        for sd, sign in (("l", -1), ("r", 1)):
            sh = (neck[0] + px * sign * self.SHOULDER[0] * k - ux * self.SHOULDER[1] * k,
                  neck[1] + py * sign * self.SHOULDER[0] * k - uy * self.SHOULDER[1] * k)
            a_sh, a_el = s[f"arm_{sd}"]
            el = self._seg("upper_arm", sh, a_sh, self.UPPER, k, sd == "l")
            wr = self._seg("forearm", el, a_sh + a_el, self.FORE, k, sd == "l")
            dx, dy = self._dir(a_sh + a_el)
            if not self._bone("hand", wr, (wr[0] + dx * 40 * k, wr[1] + dy * 40 * k), 40, k, sd == "l"):
                self._ellipse(wr[0], wr[1], r * 0.17, r * 0.17)
                self._fs(self.skin, lw)
