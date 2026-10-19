"""tools/download_model.py: the manifest covers the pinned files, and verification catches missing or changed files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import download_model  # noqa: E402

RELEASE = "mimetic_a0/mimetic-a0/"


def test_manifest_has_the_release_files_and_the_text_encoder():
    files = download_model.expected_files()
    assert set(files) == {RELEASE + f for f in ("LICENSE", "NOTICE", "PROMPTING.md", "README.md", "model.safetensors", "model.fp16.safetensors",
                                                "model_config.json")} | {"text_encoders/t5_base.safetensors"}
    assert all(len(d) == 64 for d in files.values())


def test_verify_only_reports_missing_and_changed_files(tmp_path, capsys):
    folder = tmp_path / "mimetic_a0" / "mimetic-a0"
    folder.mkdir(parents=True)
    (folder / "model_config.json").write_text("{}", encoding="utf-8")
    assert download_model.main(["--models-dir", str(tmp_path), "--weights", "fp16", "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert f"MISMATCH {RELEASE}model_config.json" in out and f"MISSING  {RELEASE}model.fp16.safetensors" in out
    assert f"{RELEASE}model.safetensors" not in out  # --weights fp16 skips the fp32 file
    assert "MISSING  text_encoders/t5_base.safetensors" in out


def test_another_t5_base_file_is_kept(tmp_path, capsys):
    (tmp_path / "text_encoders").mkdir()
    mine = tmp_path / "text_encoders" / "t5_base.safetensors"
    mine.write_bytes(b"someone else's t5")
    download_model.main(["--models-dir", str(tmp_path), "--verify-only"])
    assert "kept     text_encoders/t5_base.safetensors" in capsys.readouterr().out
    assert mine.read_bytes() == b"someone else's t5"


def test_verify_only_accepts_the_real_files(models_dir):
    assert download_model.main(["--models-dir", str(models_dir), "--verify-only"]) == 0
