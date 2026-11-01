"""Turns the source clips in demo/src into cartoons through the normal app engine."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import jobs  # noqa: E402

CLIPS = sys.argv[1:] or ["musk1", "musk2", "trump1", "trump2", "stock1"]
jobs.start()
DIRECTION = {
    "musk": "Adult man with short dark brown hair swept to the side, clean shaven, no glasses, square jaw, plain hair with no headband and no hat. Black suit, white shirt, dark grey tie. Do NOT draw a microphone, desk or table in front of him. Clean 2D cartoon.",
    "trump": "Older man with short swept-back golden blond hair, full round face, no glasses. His hair is SHORT and stops above the ears: never long hair, never a helmet of hair. Navy suit, white shirt, long red tie, small flag pin. Plain office wall behind him. Do NOT draw any table, desk, chair or other object in the foreground: the lower part of the picture must stay empty so his head, shoulders and chest are fully visible. Clean 2D cartoon.",
    "stock": "Keep the same hair, clothes and office. Clean 2D cartoon.",
}
made = {}
for name in CLIPS:
    made[name] = jobs.new_convert(str(ROOT / "demo" / "src" / f"{name}.mp4"), description=DIRECTION[name.rstrip("0123456789")],
                                  height=720, art="ai", max_people=1, motion="director")
    print("queued", name, made[name].id, flush=True)
while any(j.active for j in made.values()):
    time.sleep(10)
for name, j in made.items():
    print(name, j.id, j.status, (j.error or "")[:200], flush=True)
f = ROOT / "demo" / "src" / "jobs.json"
f.write_text(json.dumps({**json.loads(f.read_text()), **{k: v.id for k, v in made.items()}}), encoding="utf-8")
