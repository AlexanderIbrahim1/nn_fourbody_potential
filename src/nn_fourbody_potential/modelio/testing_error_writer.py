from pathlib import Path

class TestingErrorWriter:
    """Saves the testing error as a function of the epoch."""
    _savefile: Path
    
    def __init__(self, savepath: Path, filename: str = "testing_error_vs_epoch.dat") -> None:
        if not savepath.exists():
            savepath.mkdir()
            
        self._savefile = Path(savepath, filename)
        with open(self._savefile, 'w') as fout:
            fout.write("[Epoch]   [Testing Error]\n")
        
    def append(self, epoch: int, testing_error: float) -> None:
        with open(self._savefile, 'a') as fout:
            fout.write(f"{epoch: >7d} : {testing_error: .14e}\n")