#!/usr/bin/env python3
import argparse
import math
import os
import re
import subprocess
from pathlib import Path


CHUNK_RE = re.compile(r"^(?P<base>.+)\.chunk(?P<index>\d{2})of(?P<count>\d{2})$")
MAX_CHUNK_SIZE = 45 * 1024 * 1024


def valid_chunk(filename: str) -> bool:
  match = CHUNK_RE.match(filename)
  if match is None or os.path.getsize(filename) > MAX_CHUNK_SIZE:
    return False

  manifest = Path(f"{match['base']}.chunkmanifest")
  try:
    count = int(manifest.read_text().strip())
  except (FileNotFoundError, ValueError):
    return False
  return count == int(match['count']) and 1 <= int(match['index']) <= count


def lfs_files(filenames: list[str]) -> set[str]:
  if not filenames:
    return set()

  result = subprocess.run(
    ("git", "check-attr", "filter", "-z", "--stdin"),
    input="\0".join(filenames),
    check=True,
    capture_output=True,
    text=True,
  )
  fields = result.stdout.rstrip("\0").split("\0") if result.stdout else []
  return {fields[i] for i in range(0, len(fields), 3) if fields[i + 2] == "lfs"}


def check_added_large_files(filenames: list[str], max_kb: int) -> int:
  failed = False
  ignored = lfs_files(filenames)
  for filename in filenames:
    if filename in ignored or valid_chunk(filename):
      continue

    size_kb = math.ceil(os.stat(filename).st_size / 1024)
    if size_kb > max_kb:
      print(f"{filename} ({size_kb} KB) exceeds {max_kb} KB.")
      failed = True

  return int(failed)


def main() -> int:
  parser = argparse.ArgumentParser(description="Check that tracked files do not exceed a size limit.")
  parser.add_argument("filenames", nargs="*")
  parser.add_argument("--maxkb", type=int, default=500, help="maximum allowable size in KiB")
  args = parser.parse_args()
  return check_added_large_files(args.filenames, args.maxkb)


if __name__ == "__main__":
  raise SystemExit(main())
