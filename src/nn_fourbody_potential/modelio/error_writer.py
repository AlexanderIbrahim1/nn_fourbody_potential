from pathlib import Path


class ErrorWriter:
    """Saves the error as a function of the epoch."""

    _savefile: Path

    def __init__(self, savepath: Path, filename: str) -> None:
        if not savepath.exists():
            savepath.mkdir()

        self._savefile = Path(savepath, filename)
        with open(self._savefile, "w") as fout:
            fout.write("[Epoch]   [Error]\n")

    def append(self, epoch: int, error: float) -> None:
        with open(self._savefile, "a") as fout:
            fout.write(f"{epoch: >7d} : {error: .14e}\n")
