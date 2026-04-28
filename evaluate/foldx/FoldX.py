import os
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional


class FoldXSession:
    """
    Lightweight FoldX runner used by evaluate/foldx scripts.

    It creates an isolated temporary workdir and runs FoldX commands there.
    The FoldX binary path can be configured by:
    1) constructor argument `foldx_bin`
    2) env `FOLDX_BINARY`
    3) env `FOLDX_BIN`
    4) searching PATH for common executable names
    """

    def __init__(
        self,
        foldx_bin: Optional[str] = None,
        rotabase_path: Optional[str] = None,
        keep_tmp: bool = False,
    ):
        self.foldx_bin = self._resolve_foldx_bin(foldx_bin)
        self.rotabase_path = rotabase_path or os.environ.get("FOLDX_ROTABASE")
        self.keep_tmp = keep_tmp
        self._tmpdir = None

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prepare_rotabase()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tmpdir is not None and not self.keep_tmp:
            self._tmpdir.cleanup()
        self._tmpdir = None

    @property
    def workdir(self) -> str:
        if self._tmpdir is None:
            raise RuntimeError("FoldXSession is not active. Use it in a context manager.")
        return self._tmpdir.name

    def path(self, filename: str) -> str:
        return os.path.join(self.workdir, filename)

    def preprocess_data(self, input_dir: str, filename: str) -> str:
        """
        Copy input PDB into session workdir and return destination path.
        """
        src = os.path.join(input_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Input PDB not found: {src}")
        dst = self.path(filename)
        shutil.copy2(src, dst)
        return dst

    def execute_foldx(
        self,
        pdb_name: str,
        command_name: str,
        options: Optional[Iterable[str]] = None,
    ) -> subprocess.CompletedProcess:
        """
        Execute one FoldX command for a pdb file in current session.
        """
        cmd = [
            self.foldx_bin,
            f"--command={command_name}",
            f"--pdb={pdb_name}",
        ]
        if options:
            cmd.extend(list(options))

        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(
                "FoldX command failed:\n"
                f"cmd: {' '.join(cmd)}\n"
                f"return code: {out.returncode}\n"
                f"stdout:\n{out.stdout}\n"
                f"stderr:\n{out.stderr}"
            )
        return out

    @staticmethod
    def _resolve_foldx_bin(explicit_path: Optional[str]) -> str:
        if explicit_path:
            return explicit_path

        env_path = os.environ.get("FOLDX_BINARY") or os.environ.get("FOLDX_BIN")
        if env_path:
            return env_path

        for name in ("foldx", "FoldX", "foldx_linux64"):
            path = shutil.which(name)
            if path:
                return path

        raise FileNotFoundError(
            "FoldX executable not found. Set FOLDX_BINARY/FOLDX_BIN or add foldx to PATH."
        )

    def _prepare_rotabase(self):
        """
        Ensure rotabase.txt is available in workdir.

        FoldX usually works if rotabase.txt is in the working directory or
        alongside the executable. We copy it to workdir when available.
        """
        work_rotabase = self.path("rotabase.txt")
        if os.path.exists(work_rotabase):
            return

        candidates = []
        if self.rotabase_path:
            candidates.append(self.rotabase_path)

        foldx_dir = os.path.dirname(self.foldx_bin)
        candidates.append(os.path.join(foldx_dir, "rotabase.txt"))
        candidates.append(os.path.join(foldx_dir, "Rotabase.txt"))

        for cand in candidates:
            if cand and os.path.exists(cand):
                shutil.copy2(cand, work_rotabase)
                return
