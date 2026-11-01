"""Builds the Microsoft Store package from dist_store/ToonDirector.

    python packaging/build_store.py --name PACKAGE_NAME --publisher "CN=..." --display "PUBLISHER DISPLAY NAME"

The three values come from Partner Center: your app, Product identity. They must match exactly.
Needs makeappx.exe; pass --makeappx if it is not on the PATH. The Store signs the package itself.
"""
import argparse
import re
import shutil
import subprocess
import sys
import xml.sax.saxutils as xml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402

from app import config  # noqa: E402
from studio.widgets import app_icon  # noqa: E402

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
         IgnorableNamespaces="uap rescap">
  <Identity Name="{name}" Publisher="{publisher}" Version="{version}.0" ProcessorArchitecture="x64"/>
  <Properties>
    <DisplayName>ToonDirector</DisplayName>
    <PublisherDisplayName>{display}</PublisherDisplayName>
    <Logo>Assets\\StoreLogo.png</Logo>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0"/>
  </Dependencies>
  <Resources><Resource Language="en-us"/></Resources>
  <Applications>
    <Application Id="App" Executable="ToonDirector.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements DisplayName="ToonDirector" Description="Turn a topic into a finished animated video with AI."
                          BackgroundColor="transparent" Square150x150Logo="Assets\\Square150x150Logo.png"
                          Square44x44Logo="Assets\\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="Assets\\Wide310x150Logo.png"/>
      </uap:VisualElements>
    </Application>
  </Applications>
  <Capabilities>
    <Capability Name="internetClient"/>
    <rescap:Capability Name="runFullTrust"/>
  </Capabilities>
</Package>
"""


def assets(folder):
    QGuiApplication(sys.argv[:1])
    folder.mkdir(exist_ok=True)
    for name, size in (("StoreLogo", 50), ("Square44x44Logo", 44), ("Square150x150Logo", 150)):
        app_icon(size).save(str(folder / f"{name}.png"))
    wide = QImage(310, 150, QImage.Format_ARGB32)
    wide.fill(QColor("#23264F"))
    p = QPainter(wide)
    p.setRenderHint(QPainter.Antialiasing)
    p.drawImage(95, 15, app_icon(120))
    p.end()
    wide.save(str(folder / "Wide310x150Logo.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--publisher", required=True)
    ap.add_argument("--display", required=True)
    ap.add_argument("--makeappx", default="makeappx")
    ap.add_argument("--check", action="store_true", help="placeholder identity allowed, for a test build")
    a = ap.parse_args()
    if not a.check and not (config.SUPPORT_EMAIL and config.PRIVACY_URL):
        sys.exit("Fill in SUPPORT_EMAIL and PRIVACY_URL in app/config.py, rebuild dist_store, then run this again.")
    if not a.check and not re.match(r"^CN=", a.publisher):
        sys.exit("Publisher must be the full CN=... value from Partner Center.")
    src = ROOT / "dist_store" / "ToonDirector"
    if not (src / "ToonDirector.exe").exists():
        sys.exit("Build dist_store first. See store/README.md.")
    assets(src / "Assets")
    (src / "AppxManifest.xml").write_text(MANIFEST.format(
        name=xml.escape(a.name), publisher=xml.escape(a.publisher, {'"': "&quot;"}), display=xml.escape(a.display),
        version=config.VERSION), encoding="utf-8")
    out = ROOT / "dist_store" / f"ToonDirector-{config.VERSION}.msix"
    out.unlink(missing_ok=True)
    subprocess.run([a.makeappx, "pack", "/d", str(src), "/p", str(out), "/o"], check=True)
    print("Built", out, f"{out.stat().st_size / 2 ** 20:.0f} MB")


if __name__ == "__main__":
    main()
