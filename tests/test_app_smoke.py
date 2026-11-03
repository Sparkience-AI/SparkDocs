import tomllib
from pathlib import Path

APP_SOURCE = (Path(__file__).parents[1] / "code.py").read_text(encoding="utf-8")


def test_app_entry_point_compiles_without_shared_passkey() -> None:
    compile(APP_SOURCE, "code.py", "exec")
    assert "SparkDocs" in APP_SOURCE
    assert "file_uploader" in APP_SOURCE
    assert "HybridRetriever" in APP_SOURCE
    assert "build_index" in APP_SOURCE
    assert 'st.text_input("Passkey"' not in APP_SOURCE
    assert "shared_password" not in APP_SOURCE


def test_local_streamlit_defaults_are_private() -> None:
    config_path = Path(__file__).parents[1] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["server"]["address"] == "127.0.0.1"
    assert config["browser"]["gatherUsageStats"] is False
