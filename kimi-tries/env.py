"""
Carrega o arquivo .env para dentro de os.environ. Importar este
módulo (mesmo sem usar nada dele) garante que ANTHROPIC_API_KEY e
as demais variáveis estejam disponíveis ANTES de qualquer módulo
que as leia no nível de importação (ex. MODEL = os.environ.get(...)
em llm_calls.py).

python-dotenv não sobrescreve variáveis já definidas no ambiente do
shell por padrão -- então export ANTHROPIC_API_KEY=... manual continua
tendo prioridade sobre o .env, se ambos existirem.
"""

from dotenv import load_dotenv

load_dotenv()
