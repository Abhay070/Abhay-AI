"""
Python execution.

This runs model-written code on your machine. That is genuinely useful — real
data analysis, verified algorithms, actual computation instead of predicted
output — and genuinely dangerous, so it is OFF by default and must be turned on
deliberately with ENABLE_CODE_EXEC=true.

What the isolation here does and does not give you, stated honestly:

  Does:     separate process, hard timeout, no network module preloaded,
            output truncation, temp working directory.
  Does not: a real sandbox. The subprocess runs as your user and can import
            anything installed, read your files, and open sockets.

If you need real isolation, run Praxis in a container, or swap the subprocess
call below for a Docker/gVisor/Firecracker invocation. The interface stays the
same. Do not enable this on a machine holding anything you would mind losing.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import settings
from . import Tool, ToolResult, register

MAX_OUTPUT = 4000

PREAMBLE = """\
import sys, math, json, re, random, statistics, itertools, collections, datetime
"""


def run_python(code: str) -> ToolResult:
    if not settings.enable_code_exec:
        return ToolResult(False, "Code execution is disabled. Set ENABLE_CODE_EXEC=true "
                                 "to enable it, and read praxis/tools/code.py first.")
    code = str(code)
    if not code.strip():
        return ToolResult(False, "No code given.")

    with tempfile.TemporaryDirectory(prefix="praxis-exec-") as workdir:
        script = Path(workdir) / "snippet.py"
        script.write_text(PREAMBLE + "\n" + code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True,
                timeout=settings.code_exec_timeout,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False,
                              f"Timed out after {settings.code_exec_timeout}s. "
                              "Likely an infinite loop or something too slow to run here.")
        except OSError as e:
            return ToolResult(False, f"Could not start Python: {e}")

    stdout = (proc.stdout or "")[:MAX_OUTPUT]
    stderr = (proc.stderr or "")[:2000]

    if proc.returncode != 0:
        return ToolResult(False, f"Exited with code {proc.returncode}.\n{stderr}".strip(),
                          {"stdout": stdout, "stderr": stderr})
    if not stdout.strip():
        return ToolResult(True, "Ran successfully with no output. "
                                "Remember to print() what you want to see.",
                          {"stdout": ""})
    return ToolResult(True, stdout, {"stdout": stdout, "stderr": stderr})


register(Tool(
    name="run_python",
    description="Execute Python and get real output. Use for data analysis, simulations, "
                "verifying an algorithm, or any computation where you want the actual "
                "result rather than your prediction of it. print() what you want back",
    args={"code": "Python source. Standard library only unless installed"},
    run=run_python,
    icon="▶",
    requires="enable_code_exec",
    dangerous=True,
))
