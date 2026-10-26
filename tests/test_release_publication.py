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
    (tmp_path / "docs/RELEASE-NOTES.md").write_text("Version 2.0.1\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="2.0.1"\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.1.zip"
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
        "    if '--tags' not in sys.argv:\n"
        "        main = os.environ.get('CURRENT_MAIN', os.environ['GITHUB_SHA'])\n"
        "        if os.environ.get('MAIN_HEAD_FILE'): main = Path(os.environ['MAIN_HEAD_FILE']).read_text()\n"
        "        print(main + '\\trefs/heads/main')\n"
        "        sys.exit(0)\n"
        "    commit = os.environ.get('EXISTING_TAG', '')\n"
        "    if not commit: sys.exit(2)\n"
        "    print(commit + '\\trefs/tags/v2.0.1')\n"
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
        "elif args[:2] == ['release', 'create'] and os.environ.get('SUPERSEDE_DURING_UPLOAD'):\n"
        "    Path(os.environ['MAIN_HEAD_FILE']).write_text('c' * 40)\n"
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
    assert "git push origin " + "a" * 40 + ":refs/tags/v2.0.1" in events
    assert "gh release create v2.0.1" in events
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
    (directory / "dist/MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.1.zip").write_bytes(b"changed")
    result, events = run_release(release_project)
    assert result.returncode != 0
    assert not events


def test_superseded_commit_does_not_create_a_tag_or_release(release_project):
    release_project[1]["CURRENT_MAIN"] = "c" * 40
    result, events = run_release(release_project)
    assert result.returncode == 0, result.stderr
    assert "SKIPPED_RELEASE" in result.stdout
    assert "refs/heads/main" in events
    assert "git push" not in events
    assert "gh release" not in events


def test_new_main_commit_during_upload_prevents_publication(release_project):
    directory, environment = release_project
    main_file = directory / "main-head"
    main_file.write_text(environment["GITHUB_SHA"])
    environment.update(MAIN_HEAD_FILE=str(main_file), SUPERSEDE_DURING_UPLOAD="1")
    result, events = run_release(release_project)
    assert result.returncode == 0, result.stderr
    assert "SKIPPED_RELEASE" in result.stdout
    assert "gh release create" in events and "--draft " in events
    assert "gh release edit" not in events
