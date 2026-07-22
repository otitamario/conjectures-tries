"""
Carrega o arquivo .env para dentro de os.environ.
Procura o .env no mesmo diretório deste arquivo (env.py), não no
diretório de execução — assim funciona independente de onde você
rodar o script.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Diretório onde este arquivo (env.py) está localizado
THIS_DIR = Path(__file__).parent.resolve()

# Caminho absoluto para o .env
ENV_PATH = THIS_DIR / ".env"

# Carrega o .env explicitamente
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    print(f"[env] Carregado: {ENV_PATH}")
else:
    # Fallback: tenta diretório atual (para compatibilidade)
    result = load_dotenv()
    if result:
        print(f"[env] Carregado do diretório atual")
    else:
        print(f"[env] AVISO: .env não encontrado em {ENV_PATH}")