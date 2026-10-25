#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/sglang/sglc_bs_helpers.sh"

assert_eq() {
  local expected="$1" actual="$2" label="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'FAIL: %s expected=<%s> actual=<%s>\n' "${label}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

assert_eq '1 2 4 8 16 32 ' "$(cap_auto_bs_list '1 2 4 8 16 32' '')" 'empty cap'
assert_eq '1 2 4 8 16 ' "$(cap_auto_bs_list '1 2 4 8 16 32' 16)" 'cap 16'
assert_eq '1 2 4 8 12 16 ' "$(cap_auto_bs_list '1 2 4 8 16' 12)" 'add missing boundary'
assert_eq '1 2 4 8 16 ' "$(cap_auto_bs_list '1 2 4 8 16' 64)" 'do not expand theory'
assert_eq '1 ' "$(cap_auto_bs_list '4 8 16' 2)" 'keep BS1'
printf 'PASS: sglc BS cap helper\n'
