"""Turns narration into the user's own voice with OpenVoice V2 (lightweight, CPU, MIT).

Runs inside the separate voice environment (.venv-voice), called as a small program:

    python voice_clone.py --ref my_voice.wav --ckpt models/openvoice_v2 --out OUTDIR in1.wav in2.wav ...

The words and their timing stay exactly the same; only the voice colour changes, so the
lip-sync made from the original narration still fits.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="10-30 s recording of the target voice")
    ap.add_argument("--ckpt", required=True, help="folder with converter/config.json + checkpoint.pth")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--tau", type=float, default=0.3, help="lower = closer to the target voice")
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()

    import torch
    from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

    torch.set_num_threads(max(os.cpu_count() or 4, 1))

    class Converter(ToneColorConverter):
        """No audio watermark (OpenVoice's own flag for it has a bug)."""

        def __init__(self, config_path, device="cpu"):
            OpenVoiceBaseClass.__init__(self, config_path, device=device)
            self.watermark_model = None
            self.version = getattr(self.hps, "_version_", "v1")

    conv = Converter(os.path.join(a.ckpt, "converter", "config.json"), device="cpu")
    conv.load_ckpt(os.path.join(a.ckpt, "converter", "checkpoint.pth"))
    tgt_se = conv.extract_se([a.ref])
    os.makedirs(a.out, exist_ok=True)
    for i, src in enumerate(a.inputs):
        src_se = conv.extract_se([src])
        out = os.path.join(a.out, os.path.basename(src))
        conv.convert(audio_src_path=src, src_se=src_se, tgt_se=tgt_se, output_path=out, tau=a.tau)
        print(f"converted {i + 1}/{len(a.inputs)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
