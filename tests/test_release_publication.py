import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def release_project(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "script").mkdir()
    shutil.copy2(root / "script/publish_release.sh", tmp_path / "script/publish_release.sh")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/RELEASE-NOTES.md").write_text("Version 2.0.0\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="2.0.0"\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.0.zip"
    archive.write_bytes(b"verified archive")
    archive.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['RELEASE_EVENTS']).open('a') as log:\n"
        "    log.write('git ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'ls-remote':\n"
        "    commit = os.environ.get('EXISTING_TAG', '')\n"
        "    if not commit: sys.exit(2)\n"
        "    print(commit + '\\trefs/tags/v2.0.0')\n"
    )
    gh = commands / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path(os.environ['RELEASE_EVENTS']).open('a') as log:\n"
        "    log.write('gh ' + ' '.join(args) + '\\n')\n"
        "if args[:2] == ['release', 'view']:\n"
        "    if os.environ.get('EXISTING_RELEASE') != '1': sys.exit(1)\n"
        "    print(json.dumps({'isDraft': False, 'url': 'https://example.test/release'}))\n"
        "elif args[:2] == ['release', 'download']:\n"
        "    directory = Path(args[args.index('--dir') + 1])\n"
        "    for file in Path('dist').iterdir(): shutil.copy2(file, directory / file.name)\n"
    )
    git.chmod(0o755)
    gh.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=f"{commands}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        GITHUB_SHA="a" * 40,
        GITHUB_REPOSITORY="owner/cleaner",
        RELEASE_EVENTS=str(tmp_path / "events.log"),
    )
    return tmp_path, environment


def run_release(project):
    directory, environment = project
    result = subprocess.run(
        ["/bin/bash", str(directory / "script/publish_release.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    events = directory / "events.log"
    return result, events.read_text() if events.exists() else ""


def test_new_release_uploads_a_draft_then_publishes(release_project):
    result, events = run_release(release_project)
    assert result.returncode == 0, result.stderr
    assert "git push origin " + "a" * 40 + ":refs/tags/v2.0.0" in events
    assert "gh release create v2.0.0" in events
    assert "--verify-tag" in events and "--draft " in events
    assert events.index("release create") < events.index("release edit")
    assert "--draft=false" in events


def test_release_refuses_a_version_tag_from_another_commit(release_project):
    release_project[1]["EXISTING_TAG"] = "b" * 40
    result, events = run_release(release_project)
    assert result.returncode == 73
    assert "Choose a new version" in result.stderr
    assert "git push" not in events and "gh release create" not in events


def test_existing_same_commit_release_is_verified_and_unchanged(release_project):
    release_project[1].update(EXISTING_TAG="a" * 40, EXISTING_RELEASE="1")
    result, events = run_release(release_project)
    assert result.returncode == 0, result.stderr
    assert "release download" in events
    assert "release create" not in events and "release edit" not in events
    assert "kept unchanged" in result.stdout


def test_bad_archive_checksum_prevents_tag_and_release_mutation(release_project):
    directory, _ = release_project
    (directory / "dist/MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.0.zip").write_bytes(b"changed")
    result, events = run_release(release_project)
    assert result.returncode != 0
    assert not events
