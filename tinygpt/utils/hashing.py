import hashlib
from pathlib import Path


def file_sha256(
    path: str | Path,
) -> str:

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()