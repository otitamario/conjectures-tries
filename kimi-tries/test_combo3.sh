#!/bin/bash
test_query() {
  echo "=== $1 ==="
  curl -s "https://api.zbmath.org/v1/document/_search" \
    --data-urlencode "search_string=$1" \
    --data-urlencode "results_per_page=1" \
    -G | python3 -c "
import json,sys
d = json.load(sys.stdin)
r = d.get('result')
print('RESULTADOS:', len(r) if r else 0, '| status_code:', d.get('status',{}).get('status_code'))
"
  echo
}

# intervalo de ano bem mais amplo, só com 1 MSC
test_query "(cc:05C*) AND (py:1900-2024)"

# 9 termos MSC, sem filtro de ano
test_query "((cc:05A*) OR (cc:05C*) OR (cc:05E*) OR (cc:11A*) OR (cc:11B*) OR (cc:11D*) OR (cc:11N*) OR (cc:11P*) OR (cc:11T*))"

# 3 termos MSC + ano amplo
test_query "((cc:05A*) OR (cc:05C*) OR (cc:05E*)) AND (py:1900-2024)"

# todos os 9 termos + ano amplo (a query que falhou)
test_query "((cc:05A*) OR (cc:05C*) OR (cc:05E*) OR (cc:11A*) OR (cc:11B*) OR (cc:11D*) OR (cc:11N*) OR (cc:11P*) OR (cc:11T*)) AND (py:1900-2024)"