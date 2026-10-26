from pathlib import Path
from unittest.mock import Mock

import pytest

from watermark_cleaner.cli import main


def test_cli_help_does_not_initialize_an_engine(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "--engine {e2fgvi,propainter}" in capsys.readouterr().out


def test_cli_regions_and_engine_reach_shared_pipeline(tmp_path, monkeypatch, capsys):
    from watermark_cleaner import core

    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source identity")
    region_file = tmp_path / "regions.json"
    region_file.write_text('[{"bbox":[0.1,0.1,0.3,0.3],"start_frame":2,"end_frame":4}]')
    cleaner = Mock()
    factory = Mock(return_value=cleaner)
    monkeypatch.setattr(core, "WatermarkCleaner", factory)
    assert (
        main(
            [
                "-i",
                str(source),
                "-o",
                str(output),
                "--engine",
                "propainter",
                "--regions",
                str(region_file),
                "--watermark-text",
                "InVideo",
            ]
        )
        == 0
    )
    factory.assert_called_once_with("propainter")
    args, kwargs = cleaner.run.call_args
    assert args == (source, output)
    assert kwargs["watermark_regions"][0]["start_frame"] == 2
    assert kwargs["watermark_text"] == "InVideo"
    assert source.read_bytes() == b"source identity"
    assert f"Completed: {output}" in capsys.readouterr().out


def test_cli_reports_failure_without_completion(tmp_path, monkeypatch, capsys):
    from watermark_cleaner import core

    source = tmp_path / "source.mp4"
    source.touch()
    cleaner = Mock()
    cleaner.run.side_effect = ValueError("No matching watermark was confirmed.")
    monkeypatch.setattr(core, "WatermarkCleaner", Mock(return_value=cleaner))
    assert main(["-i", str(source), "-o", str(tmp_path / "out.mp4")]) == 1
    result = capsys.readouterr()
    assert "No matching watermark" in result.err
    assert "Completed" not in result.out


def test_cli_refuses_existing_output_before_processing(tmp_path, capsys):
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source")
    output.write_bytes(b"previous output")
    assert main(["-i", str(source), "-o", str(output)]) == 1
    assert output.read_bytes() == b"previous output"
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_cli_empty_regions_cannot_trigger_an_unrequested_scan(tmp_path, capsys):
    source, regions = tmp_path / "source.mp4", tmp_path / "regions.json"
    source.touch()
    regions.write_text("[]")
    assert (
        main(["-i", str(source), "-o", str(tmp_path / "out.mp4"), "--regions", str(regions)]) == 1
    )
    assert "nonempty JSON list" in capsys.readouterr().err


def test_cli_batch_uses_separate_non_overwriting_outputs(tmp_path, monkeypatch):
    from watermark_cleaner import core

    sources, outputs = tmp_path / "sources", tmp_path / "outputs"
    sources.mkdir()
    for name in ["first.mp4", "second.MOV", "ignored.txt"]:
        (sources / name).touch()
    cleaner = Mock()
    monkeypatch.setattr(core, "WatermarkCleaner", Mock(return_value=cleaner))
    assert main(["-i", str(sources), "-o", str(outputs), "--quiet"]) == 0
    assert cleaner.run.call_count == 2
    destinations = [call.args[1] for call in cleaner.run.call_args_list]
    assert len(set(destinations)) == 2
    assert all(Path(path).parent == outputs and path.suffix == ".mp4" for path in destinations)
