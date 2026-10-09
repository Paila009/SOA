"""Publish the canonical customer workspace, not a separate marketing showcase.

Only Firebase's public web configuration is copied. Provider secrets, local
weights, databases and research outputs are never part of the static artifact.
"""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build(destination=None):
    target = Path(destination) if destination else ROOT / "dist" / "pages"
    target.mkdir(parents=True, exist_ok=True)
    assets = target / "assets"
    assets.mkdir(exist_ok=True)
    source = ROOT / "customer" / "web"
    # Deliberate allowlist: never recursively copy the app or environment files.
    for name in ("app.js", "browser-runtime.js", "style.css", "logo.svg"):
        shutil.copyfile(source / name, assets / name)
    firebase = json.loads((ROOT / "pages" / "firebase-config.json").read_text(encoding="utf-8"))
    required = {"apiKey", "authDomain", "projectId", "appId"}
    if set(firebase) != required or not all(isinstance(v, str) and v for v in firebase.values()):
        raise ValueError("Pages needs the four public Firebase web configuration fields.")
    deployment = {"mode": "browser", "firebase": firebase}
    (target / "deployment-config.js").write_text(
        "globalThis.GROUNDED_DEPLOYMENT = " + json.dumps(deployment) + ";\n", encoding="utf-8")
    html = (source / "index.html").read_text(encoding="utf-8")
    marker = '<script type="module" src="./assets/app.js"></script>'
    if marker not in html:
        raise ValueError("The customer module script must use a relative asset path.")
    html = html.replace(marker, '<script src="./deployment-config.js"></script>\n'
                        '  <script src="./assets/browser-runtime.js"></script>\n  ' + marker)
    (target / "index.html").write_text(html, encoding="utf-8")
    (target / ".nojekyll").write_text("", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
