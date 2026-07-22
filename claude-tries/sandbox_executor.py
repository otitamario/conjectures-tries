"""
Roda o código gerado pelo Prompt 3 em um subprocesso isolado
(venv/container dedicado), com timeout. Isso é execução real,
não uma chamada de LLM — é aqui que "PASS/FAIL" vira fato, não opinião.
"""

import env  # noqa: F401  (carrega .env antes do os.environ.get abaixo)
import subprocess
import tempfile
import os
from schemas import ExecutionResult

DEFAULT_TIMEOUT_SECONDS = 30
PYTHON_BIN = os.environ.get("SANDBOX_PYTHON", ".venv/bin/python")


def run_candidate_code(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> ExecutionResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [PYTHON_BIN, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            passed=(proc.returncode == 0),
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            passed=False,
            stdout=e.stdout or "",
            stderr=f"TIMEOUT após {timeout}s (possível busca infinita ou n_max grande demais)",
            exit_code=-1,
        )
    finally:
        os.unlink(script_path)
