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

test_query "(cc:05C*) OR (cc:11B*)"
test_query "(cc:05C*) OR (cc:11B*) AND (py:2020-2024)"
test_query "((cc:05C*) OR (cc:11B*)) AND (py:2020-2024)"