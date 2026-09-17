"""Download a small public-domain classical MIDI dataset.

Default source: *The Mutopia Project* (https://www.mutopiaproject.org) - an
archive of freely licensed musical scores. All music on Mutopia is either in the
public domain or released under Creative Commons "free cultural works" licenses,
so it is safe to use and redistribute.

Two modes
---------
1. Default: download a curated list of ~16 well-known pieces (Bach, Chopin,
   Beethoven, Mozart, Joplin, Pachelbel) - a small, single-instrument-friendly
   corpus that trains quickly on CPU.
2. ``--composer BachJS --limit 50``: crawl the Mutopia FTP index tree of a
   composer directory and download the first N MIDI files found. Useful for
   building a bigger corpus.

Custom MIDI files you already have can simply be copied into ``data/raw`` - the
preprocessing step picks up every ``*.mid`` file there.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MUTOPIA_BASE = "https://www.mutopiaproject.org/ftp"

# Curated public-domain pieces with their Mutopia FTP paths (verified paths).
CURATED_PIECES = [
    "BachJS/BWV846/wtk1-prelude1/wtk1-prelude1.mid",
    "BachJS/BWV846/wtk1-fugue1/wtk1-fugue1.mid",
    "BachJS/BWV847/fuga2/fuga2.mid",
    "MozartWA/AveverumM/AveverumM.mid",
    "ChopinFF/O9/chopin_nocturne_op9_n2/chopin_nocturne_op9_n2.mid",
    "ChopinFF/O9/nocturne_in_b-flat_minor/nocturne_in_b-flat_minor.mid",
    "ChopinFF/O28/Chop-28-1/Chop-28-1.mid",
    "ChopinFF/O28/Chop-28-4/Chop-28-4.mid",
    "ChopinFF/O28/Chop-28-15/Chop-28-15.mid",
    "BeethovenLv/O49/LVB_Sonate_49no1_1/LVB_Sonate_49no1_1.mid",
    "BeethovenLv/O49/LVB_Sonate_49no2_1/LVB_Sonate_49no2_1.mid",
    "JoplinS/entertainer/entertainer.mid",
    "JoplinS/maple/maple.mid",
    "JoplinS/bethena/bethena.mid",
    "PachelbelJ/CanonInD/CanonInD.mid",
    "PachelbelJ/PachelbelGigue/PachelbelGigue.mid",
]

USER_AGENT = "MusicGenerationWithAI/1.0 (educational use)"


def build_url(rel_path: str) -> str:
    return f"{MUTOPIA_BASE}/{rel_path.strip('/')}"


def sanitise_filename(url: str) -> str:
    """Build a unique, readable filename from a Mutopia URL path."""
    segments = [s for s in urllib.parse.urlparse(url).path.split("/") if s]
    # Drop the leading "ftp" segment.
    rel = segments[segments.index("ftp") + 1 :] if "ftp" in segments else segments
    return "_".join(rel)


def download(url: str, target: pathlib.Path, timeout: int) -> bool:
    """Download a single file; returns True on success."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            print(f"  [warn] empty file, skipping {url}")
            return False
        target.write_bytes(data)
        return True
    except urllib.error.HTTPError as exc:
        print(f"  [skip] {url} -> HTTP {exc.code}")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [skip] {url} -> {exc}")
        return False


def download_curated(out_dir: pathlib.Path, timeout: int) -> int:
    """Download the curated default piece list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CURATED_PIECES)} curated public-domain pieces "
          f"from {MUTOPIA_BASE}")
    ok = 0
    for rel in CURATED_PIECES:
        url = build_url(rel)
        name = sanitise_filename(url)
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  [exists] {name}")
            ok += 1
            continue
        print(f"  [fetch ] {url}")
        if download(url, target, timeout):
            print(f"           -> {target} ({target.stat().st_size:,} bytes)")
            ok += 1
        time.sleep(0.3)
    print(f"Done: {ok}/{len(CURATED_PIECES)} files downloaded.")
    return ok


# ---------------------------------------------------------------------------
# Composer crawl mode
# ---------------------------------------------------------------------------

def fetch_html(url: str, timeout: int) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        return html
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def crawl_index(html: str, page_url: str) -> tuple[list[str], list[str]]:
    """Parse a directory index into (child_dirs, midi_urls)."""
    import re

    child_dirs, midi_urls = [], []
    for href in re.findall(r'href="([^"]+)"', html):
        if not href or href.startswith(("#", "?", "mailto")):
            continue
        resolved = urllib.parse.urljoin(page_url, href)
        if resolved.endswith("/"):
            child_dirs.append(resolved)
        elif resolved.lower().endswith((".mid", ".midi")) and ".rdf" not in resolved:
            midi_urls.append(resolved)
    return child_dirs, midi_urls


def crawl_composer(
    composer: str, out_dir: pathlib.Path, limit: int, timeout: int
) -> int:
    """BFS-crawl a Mutopia composer directory and download MIDI files."""
    base = f"{MUTOPIA_BASE}/{composer}/"
    queue = [(base, 0)]
    visited: set[str] = set()
    found_midi: list[str] = []

    print(f"Crawling {base} (bounded to {limit} files)...")
    while queue and len(found_midi) < limit:
        url, depth = queue.pop(0)
        if url in visited or depth > 4:
            continue
        visited.add(url)
        html = fetch_html(url, timeout)
        if html is None:
            continue
        child_dirs, midi_urls = crawl_index(html, url)
        for midi in midi_urls:
            if midi not in found_midi:
                found_midi.append(midi)
                print(f"  [found] {midi}")
                if len(found_midi) >= limit:
                    break
        if len(found_midi) >= limit:
            break
        # Only descend deeper inside this composer's branch of the tree.
        for child in child_dirs:
            if child.startswith(base) and child not in visited:
                queue.append((child, depth + 1))
        time.sleep(0.2)

    if not found_midi:
        print(f"Found no MIDI files under {base}. Exiting.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading up to {limit} MIDI files...")
    ok = 0
    for url in found_midi[:limit]:
        name = sanitise_filename(url)
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  [exists] {name}")
            ok += 1
            continue
        if download(url, target, timeout):
            print(f"           -> {target} ({target.stat().st_size:,} bytes)")
            ok += 1
        time.sleep(0.2)
    print(f"Done: {ok}/{len(found_midi[:limit])} files downloaded.")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a public-domain classical MIDI dataset (Mutopia)."
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("data/raw"),
        help="directory to save MIDI files into (default: data/raw)",
    )
    parser.add_argument(
        "--urls", type=pathlib.Path, default=None,
        help="text file with one direct .mid URL per line (overrides curated list)",
    )
    parser.add_argument(
        "--composer", default=None,
        help="crawl the Mutopia FTP tree of a composer dir, e.g. BachJS or ChopinFF",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="max files to download when crawling (default: 50)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="HTTP timeout in seconds (default: 30)",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir

    if args.urls:
        if not args.urls.exists():
            print(f"URL list file not found: {args.urls}")
            return 1
        lines = [ln.strip() for ln in args.urls.read_text().splitlines() if ln.strip()]
        ok = 0
        for line in lines:
            name = sanitise_filename(line)
            print(f"  [fetch ] {line}")
            if download(line, out_dir / name, args.timeout):
                ok += 1
        print(f"Done: {ok}/{len(lines)} files downloaded from URL list.")
        return 0 if ok > 0 else 1

    if args.composer:
        count = crawl_composer(args.composer, out_dir, args.limit, args.timeout)
        return 0 if count > 0 else 1

    count = download_curated(out_dir, args.timeout)
    if count == 0:
        print(
            "\nNo files were downloaded. Check your internet connection or use "
            "`--urls FILE` / `--composer NAME`."
        )
        return 1

    print("\nNext step: python scripts/preprocess.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
