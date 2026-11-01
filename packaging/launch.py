"""Entry point of the installed app. `--selftest report.txt` checks the packaged engine, and
`--selftest report.txt video` also makes one short video, without opening the window."""
import sys


def selftest(report, video):
    import importlib
    import time
    import traceback
    with open(report, "w", encoding="utf-8", buffering=1) as out:
        from app import config
        out.write(f"store edition: {config.STORE}\n")
        for name in ("cv2", "cairo", "resvg_py", "edge_tts", "mediapipe", "faster_whisper", "ctranslate2", "openai",
                     "PySide6.QtMultimedia", "PySide6.QtSvg", "app.pipeline", "app.story", "app.animator", "studio.main"):
            try:
                importlib.import_module(name)
                out.write(f"ok   {name}\n")
            except Exception:
                out.write(f"FAIL {name}\n{traceback.format_exc()}\n")
        if not video:
            return 0
        from app import jobs, providers
        out.write(f"providers: {[p['name'] for p in providers.ready()]}\n")
        jobs.start()
        job = jobs.new_story("topic", topic="Why the Persian Gulf is so salty", minutes=0.5, height=720)
        while job.active:
            time.sleep(2)
        out.write(f"video: {job.status} {job.error} {job.output.exists()} {job.logs[-1][:120]}\n")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        sys.exit(selftest(sys.argv[i + 1], "video" in sys.argv))
    from studio.main import main
    sys.exit(main())
