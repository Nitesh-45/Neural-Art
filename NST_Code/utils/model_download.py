import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


def ensure_model_file(path, url=None, url_setting="MODEL_URL"):
    destination = Path(path)
    if destination.is_file():
        return destination

    url = (url or os.environ.get(url_setting, "")).strip()
    if not url:
        raise FileNotFoundError(
            f"Model file not found at {destination}. Set {url_setting} to its direct download URL."
        )

    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"{url_setting} must contain a direct HTTP(S) download URL.")

    if parsed_url.netloc.lower() == "drive.google.com":
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) >= 3 and path_parts[:2] == ["file", "d"]:
            url = f"https://drive.google.com/uc?export=download&id={path_parts[2]}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".download",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
            if response.headers.get_content_type() == "text/html":
                raise ValueError(
                    f"{url_setting} returned a web page instead of a model file. "
                    "Use a direct download URL with public file access."
                )
            shutil.copyfileobj(response, output)

        if temporary_path.stat().st_size == 0:
            raise ValueError(f"Downloaded model from {url_setting} is empty.")

        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return destination
