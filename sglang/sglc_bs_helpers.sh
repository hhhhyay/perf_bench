#!/usr/bin/env bash

# 限制自动理论 BS 扫描范围；固定 BS/并发模式不会调用此函数。
cap_auto_bs_list() {
  local auto_bs_list="$1"
  local max_test_bs="$2"

  local bs theory_max=0 effective_cap
  for bs in ${auto_bs_list}; do
    if [[ "${bs}" =~ ^[0-9]+$ ]] && (( bs > theory_max )); then
      theory_max="${bs}"
    fi
  done

  if (( theory_max < 1 )); then
    printf '\n'
    return 0
  fi

  if [[ -z "${max_test_bs}" ]]; then
    effective_cap="${theory_max}"
  else
    effective_cap="${max_test_bs}"
    if (( effective_cap > theory_max )); then
      effective_cap="${theory_max}"
    fi
  fi

  {
    for bs in ${auto_bs_list}; do
      if [[ "${bs}" =~ ^[0-9]+$ ]] && (( bs <= effective_cap )); then
        printf '%s\n' "${bs}"
      fi
    done
    if [[ -n "${max_test_bs}" ]] && (( max_test_bs <= theory_max )); then
      printf '%s\n' "${max_test_bs}"
    fi
    # 即使用户上限低于第一个基础点，也保留最小可运行的 BS1。
    printf '1\n'
  } | sort -n -u | tr '\n' ' '
}
