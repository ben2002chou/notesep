"""Offer the serving Python/Studio source without uploads, models or audio assets."""

from functools import lru_cache
from pathlib import Path
import zipfile
import io


@lru_cache(maxsize=1)
def build_source_archive():
    package = Path(__file__).resolve().parent
    entries = {}
    for root in (package, package / "models"):
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Serving source package missing or symlinked")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Serving source must not contain symlinks")
            relative = path.relative_to(root)
            if not path.is_file() or "__pycache__" in relative.parts:
                continue
            source = path.suffix in {".py", ".js", ".mjs", ".css", ".html"}
            notice = relative.parts[0] == "legal" or "LICENSE" in path.name
            if source or notice:
                entries[root.name + "/" + str(relative)] = path.read_bytes()
    if "notesep/legal/LICENSE" not in entries:
        raise ValueError("Complete serving-code license missing")
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(entries.items()):
            info = zipfile.ZipInfo(
                "NoteSepSource/" + name, date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, body)
    return result.getvalue()
