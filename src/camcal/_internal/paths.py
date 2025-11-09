from pathlib import Path


def repo_root() -> Path:
    pth = Path(__file__)

    while not (pth / ".git").exists():
        pth = pth.parent

    return pth
