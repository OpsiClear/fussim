#!/usr/bin/env python3
"""
Generate PEP 503 compliant Simple Repository API index pages for pip installation.

This script fetches wheel files from GitHub releases and generates HTML index pages
that can be served via GitHub Pages for pip installation with --extra-index-url.

Usage:
    pip install fussim --extra-index-url https://opsiclear.github.io/fussim/whl/

Or for a specific PyTorch/CUDA version:
    pip install fussim --extra-index-url https://opsiclear.github.io/fussim/whl/pt25cu124/
"""

import argparse
import hashlib
import os
import re
import urllib.request
from pathlib import Path

import requests
from jinja2 import Template


def get_github_repo():
    """Get repository from environment or raise error."""
    repo = os.getenv("GITHUB_REPOSITORY")
    if not repo:
        raise ValueError("GITHUB_REPOSITORY environment variable not set")
    return repo


def download_and_hash(url: str, headers: dict = None) -> str:
    """Download file and compute SHA256 hash."""
    req = urllib.request.Request(url)
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            hash_obj = hashlib.sha256()
            while chunk := response.read(8192):
                hash_obj.update(chunk)
            return hash_obj.hexdigest()
    except Exception as e:
        print(f"Warning: Could not compute hash for {url}: {e}")
        return ""


def list_wheel_files(repo: str) -> list[dict]:
    """Fetch all wheel files from GitHub releases."""
    releases_url = f"https://api.github.com/repos/{repo}/releases"

    headers = {}
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    response = requests.get(releases_url, headers=headers)
    response.raise_for_status()
    releases = response.json()

    wheel_files = []
    wheel_pattern = re.compile(
        r"^(?P<name>[\w\d_]+)-"
        r"(?P<version>[\d.]+)"
        r"(?P<local>\+[\w\d.]+)?-"
        r"(?P<python_tag>[\w]+)-"
        r"(?P<abi_tag>[\w]+)-"
        r"(?P<platform_tag>[\w]+)\.whl$"
    )

    for release in releases:
        for asset in release.get("assets", []):
            filename = asset["name"]
            if not filename.endswith(".whl"):
                continue

            match = wheel_pattern.match(filename)
            if not match:
                print(f"Warning: Could not parse wheel filename: {filename}")
                continue

            local_version = match.group("local")
            if local_version:
                local_version = local_version.lstrip("+")

            # Compute SHA256 hash from download URL
            download_url = asset["browser_download_url"]
            print(f"Computing hash for {filename}...")
            sha256 = download_and_hash(download_url, headers)

            wheel_files.append(
                {
                    "release_name": release["name"],
                    "wheel_name": filename,
                    "download_url": download_url,
                    "package_name": match.group("name").replace("_", "-"),
                    "version": match.group("version"),
                    "local_version": local_version,
                    "sha256": sha256,
                }
            )

    return wheel_files


def generate_index_html(wheels: list[dict], repo: str, outdir: Path):
    """Generate PEP 503 compliant index HTML files."""

    # Template for package listing (shows all wheels for a package)
    package_template = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Links for {{ package_name }}</title>
</head>
<body>
    <h1>Links for {{ package_name }}</h1>
    {% for wheel in wheels %}
    <a href="{{ wheel.download_url }}{% if wheel.sha256 %}#sha256={{ wheel.sha256 }}{% endif %}">{{ wheel.wheel_name }}</a><br>
    {% endfor %}
</body>
</html>
""")

    # Template for root index (lists all packages)
    root_template = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Simple Index</title>
</head>
<body>
    <h1>Simple Index</h1>
    {% for package_name in packages %}
    <a href="{{ package_name }}/">{{ package_name }}</a><br>
    {% endfor %}
</body>
</html>
""")

    # Group wheels by package name
    packages = {}
    for wheel in wheels:
        pkg = wheel["package_name"]
        packages.setdefault(pkg, []).append(wheel)

    # Generate root index
    outdir.mkdir(parents=True, exist_ok=True)
    root_html = root_template.render(packages=sorted(packages.keys()))
    (outdir / "index.html").write_text(root_html)
    print(f"Generated: {outdir}/index.html")

    # Generate per-package index
    for package_name, pkg_wheels in packages.items():
        pkg_dir = outdir / package_name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        pkg_html = package_template.render(
            package_name=package_name, wheels=sorted(pkg_wheels, key=lambda w: w["wheel_name"])
        )
        (pkg_dir / "index.html").write_text(pkg_html)
        print(f"Generated: {pkg_dir}/index.html ({len(pkg_wheels)} wheels)")

    # Generate per-local-version indexes (e.g., pt25cu124)
    local_versions = {}
    for wheel in wheels:
        lv = wheel.get("local_version")
        if lv:
            local_versions.setdefault(lv, []).append(wheel)

    for local_version, lv_wheels in local_versions.items():
        lv_dir = outdir / local_version
        lv_dir.mkdir(parents=True, exist_ok=True)

        # Group by package within local version
        lv_packages = {}
        for wheel in lv_wheels:
            pkg = wheel["package_name"]
            lv_packages.setdefault(pkg, []).append(wheel)

        # Root index for this local version
        lv_root_html = root_template.render(packages=sorted(lv_packages.keys()))
        (lv_dir / "index.html").write_text(lv_root_html)

        # Package indexes for this local version
        for package_name, pkg_wheels in lv_packages.items():
            pkg_dir = lv_dir / package_name
            pkg_dir.mkdir(parents=True, exist_ok=True)

            pkg_html = package_template.render(
                package_name=package_name, wheels=sorted(pkg_wheels, key=lambda w: w["wheel_name"])
            )
            (pkg_dir / "index.html").write_text(pkg_html)

        print(f"Generated: {lv_dir}/ ({len(lv_wheels)} wheels)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pip-compatible index pages from GitHub releases"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("."), help="Output directory for index pages"
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip computing SHA256 hashes (faster but less secure)",
    )
    args = parser.parse_args()

    repo = get_github_repo()
    print(f"Fetching wheels from: {repo}")

    wheels = list_wheel_files(repo)
    print(f"Found {len(wheels)} wheel files")

    if wheels:
        generate_index_html(wheels, repo, args.outdir)
        print("\nInstallation instructions:")
        print("  pip install fussim --extra-index-url https://opsiclear.github.io/fussim/whl/")
    else:
        print("No wheels found in releases")


if __name__ == "__main__":
    main()
