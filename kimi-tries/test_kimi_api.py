#!/usr/bin/env python3
"""Teste direto da conexão com API Kimi."""

import os
from openai import OpenAI

# Força recarregar .env
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

key = os.environ.get("KIMI_API_KEY")
print("=" * 60)
print("DIAGNÓSTICO DA API KIMI")
print("=" * 60)
print(f"\nAPI Key encontrada: {'SIM' if key else 'NÃO'}")
if key:
    print(f"Prefixo: {key[:10]}...")
    print(f"Tamanho: {len(key)} caracteres")
    print(f"Termina com: ...{key[-4:]}")

# Teste 1: Listar modelos
print("\n--- Teste 1: Listar modelos ---")
for base_url in ["https://api.moonshot.cn/v1", "https://api.moonshot.ai/v1"]:
    print(f"\nBase URL: {base_url}")
    try:
        client = OpenAI(api_key=key, base_url=base_url)
        models = client.models.list()
        print(f"✓ CONECTADO! Modelos disponíveis:")
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as e:
        print(f"✗ Falha: {type(e).__name__}: {e}")

# Teste 2: Chat simples
print("\n--- Teste 2: Chat simples ---")
try:
    client = OpenAI(api_key=key, base_url="https://api.moonshot.cn/v1")
    response = client.chat.completions.create(
        model="kimi-k2-5",
        messages=[{"role": "user", "content": "Diga 'API funcionando' em português"}],
        max_tokens=50,
    )
    print(f"✓ Resposta: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Falha: {type(e).__name__}: {e}")

print("\n" + "=" * 60)