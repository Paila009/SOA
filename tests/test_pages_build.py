import json
from pathlib import Path

from scripts.build_pages import build


def test_pages_build_contains_actual_workspace_not_showcase(tmp_path):
    target = build(tmp_path / "published")
    html = (target / "index.html").read_text(encoding="utf-8")
    assert 'id="google-login"' in html
    assert 'id="models-button"' in html
    assert 'id="composer"' in html
    assert 'id="answer-insights"' in html
    assert 'src="./assets/browser-runtime.js"' in html
    assert 'src="./deployment-config.js"' in html
    assert 'src="/assets/' not in html
    assert "Run the full app locally" not in html
    config = (target / "deployment-config.js").read_text(encoding="utf-8")
    deployment = json.loads(config.split(" = ", 1)[1].rstrip(";\n"))
    assert deployment["mode"] == "browser"
    assert deployment["firebase"]["projectId"] == "grounded-583ef"
    assert set(deployment["firebase"]) == {"apiKey", "authDomain", "projectId", "appId"}
    files = {str(path.relative_to(target)).replace("\\", "/") for path in target.rglob("*") if path.is_file()}
    assert files == {"index.html", "deployment-config.js", ".nojekyll", "assets/app.js",
                     "assets/browser-runtime.js", "assets/style.css", "assets/logo.svg"}
