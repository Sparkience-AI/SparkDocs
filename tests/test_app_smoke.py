import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_shared_passkey() -> None:
    app = AppTest.from_file(str(Path(__file__).parents[1] / "code.py")).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "SparkDocs"
    assert all("passkey" not in item.label.lower() for item in app.text_input)


def test_local_streamlit_defaults_are_private() -> None:
    config_path = Path(__file__).parents[1] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["server"]["address"] == "127.0.0.1"
    assert config["browser"]["gatherUsageStats"] is False
