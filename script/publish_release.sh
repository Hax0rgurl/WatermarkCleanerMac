#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${GITHUB_SHA:?The verified workflow commit is required.}"
: "${GITHUB_REPOSITORY:?The release repository is required.}"
VERSION="$(python3 -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: Unsupported release version: $VERSION" >&2
  exit 65
fi
TAG="v$VERSION"
ZIP_NAME="MAC-WATERMARK-CLEANER-macOS-arm64-v$VERSION.zip"
CHECKSUM_NAME="$ZIP_NAME.sha256"
NOTES="$ROOT_DIR/docs/RELEASE-NOTES.md"
[[ -f "$ROOT_DIR/dist/$ZIP_NAME" && -f "$ROOT_DIR/dist/$CHECKSUM_NAME" && -f "$NOTES" ]] || {
  echo "ERROR: The successful build's ZIP, checksum, or release notes are missing." >&2
  exit 66
}
(
  cd "$ROOT_DIR/dist"
  shasum -a 256 -c "$CHECKSUM_NAME"
)

tag_refs=""
if tag_refs="$(git ls-remote --exit-code --tags origin "refs/tags/$TAG" "refs/tags/$TAG^{}")"; then
  tag_commit="$(printf '%s\n' "$tag_refs" | awk '$2 ~ /\^\{\}$/ {print $1}')"
  if [[ -z "$tag_commit" ]]; then tag_commit="$(printf '%s\n' "$tag_refs" | awk 'NR == 1 {print $1}')"; fi
  if [[ "$tag_commit" != "$GITHUB_SHA" ]]; then
    echo "ERROR: $TAG already identifies $tag_commit, not this verified commit $GITHUB_SHA. Choose a new version." >&2
    exit 73
  fi
else
  result=$?
  if [[ "$result" != 2 ]]; then
    echo "ERROR: Could not inspect the existing release tag." >&2
    exit "$result"
  fi
  # This push is intentionally not forced. A concurrent different tag fails safely.
  git push origin "$GITHUB_SHA:refs/tags/$TAG"
fi

release_details="$(mktemp)"
download_dir="$(mktemp -d)"
trap 'rm -f "$release_details"; rm -rf "$download_dir"' EXIT
if gh release view "$TAG" --repo "$GITHUB_REPOSITORY" --json isDraft,url > "$release_details"; then
  gh release download "$TAG" --repo "$GITHUB_REPOSITORY" --dir "$download_dir" \
    --pattern "$ZIP_NAME" --pattern "$CHECKSUM_NAME"
  (
    cd "$download_dir"
    shasum -a 256 -c "$CHECKSUM_NAME"
  )
  is_draft="$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1]))["isDraft"]).lower())' "$release_details")"
  if [[ "$is_draft" == true ]]; then
    gh release edit "$TAG" --repo "$GITHUB_REPOSITORY" --draft=false
  else
    echo "Existing release $TAG already belongs to this commit and its assets verify. It was kept unchanged."
  fi
else
  gh release create "$TAG" --repo "$GITHUB_REPOSITORY" --verify-tag \
    --target "$GITHUB_SHA" --title "MAC WATERMARK CLEANER $VERSION" \
    --notes-file "$NOTES" --draft \
    "$ROOT_DIR/dist/$ZIP_NAME" "$ROOT_DIR/dist/$CHECKSUM_NAME"
  gh release edit "$TAG" --repo "$GITHUB_REPOSITORY" --draft=false
fi
