#!/bin/bash
# Arena 작업의 status / log / result 를 내려받아 파일로 저장하고 요약을 만든다.
#
#   scripts/rngd_fetch.sh            # 마지막으로 제출한 작업 (target/rngd/last_job_id) 또는 내 최신 작업
#   scripts/rngd_fetch.sh 31463      # 특정 작업
#
# 산출물 (RNGD_OUT_DIR, 기본 target/rngd/<id>/):
#   status.json   furiosa-arena status <id>
#   log.txt       furiosa-arena logs <id>     (stdout/stderr 전체)
#   result.txt    furiosa-arena result <id>   (수집된 결과)
#   summary.json  커널별 PASS/FAIL 과 cycles  (Slack 등 연동용)
#   summary.md    같은 내용의 마크다운 표      (stdout 에도 출력)
#
# 요구: FURIOSA_ARENA_URL 과 로그인(또는 FURIOSA_ARENA_TOKEN).
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"

OUT_ROOT="${RNGD_OUT_DIR:-target/rngd}"
LAST_ID_FILE="$OUT_ROOT/last_job_id"

job="${1:-${RNGD_JOB_ID:-}}"
if [ -z "$job" ] && [ -f "$LAST_ID_FILE" ]; then
    job="$(tr -d '[:space:]' < "$LAST_ID_FILE")"
fi
if [ -z "$job" ]; then
    # `list --mine` 의 첫 데이터 행 = 가장 최근 작업
    job="$(furiosa-arena list --mine 2>/dev/null | awk 'NR > 1 && $1 ~ /^[0-9]+$/ { print $1; exit }')"
fi
if [ -z "$job" ]; then
    echo "rngd_fetch.sh: no job id (pass one, set RNGD_JOB_ID, or submit with rngd_test.sh first)" >&2
    exit 2
fi

out="$OUT_ROOT/$job"
mkdir -p "$out"

furiosa-arena status "$job" > "$out/status.json"
furiosa-arena logs   "$job" > "$out/log.txt"    2>&1 || true
furiosa-arena result "$job" > "$out/result.txt" 2>&1 || true

python3 - "$out" <<'EOF'
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
status = json.loads((out / "status.json").read_text())
result = (out / "result.txt").read_text(errors="replace")

kernels, current = [], None
for line in result.splitlines():
    m = re.match(r"^==> (\S+)", line)
    if m:
        current = {"name": m.group(1), "checks": [], "cycles": None}
        kernels.append(current)
        continue
    if current is None:
        continue
    m = re.match(r"^\[(.+?)\s*\].*?max\|Δ\|=\s*([0-9.]+).*?->\s*(PASS|FAIL)", line)
    if m:
        current["checks"].append({"label": m.group(1).strip(), "max_abs_diff": float(m.group(2)), "pass": m.group(3) == "PASS"})
        continue
    m = re.match(r"^\s*cycles=(\d+)", line)
    if m:
        current["cycles"] = int(m.group(1))

for k in kernels:
    k["pass"] = bool(k["checks"]) and all(c["pass"] for c in k["checks"])

summary = {
    "job_id": status["id"],
    "name": status.get("name"),
    "status": status.get("status"),
    "exit_code": status.get("exit_code"),
    "duration_sec": status.get("duration_sec"),
    "finished_at": status.get("finished_at"),
    "all_passed": status.get("exit_code") == 0 and bool(kernels) and all(k["pass"] for k in kernels),
    "kernels": kernels,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

verdict = "PASS" if summary["all_passed"] else "FAIL"
lines = [
    f"## RNGD job {summary['job_id']} ({summary['name']}) — {summary['status']}, exit {summary['exit_code']}, {verdict}",
    "",
    "| kernel | accuracy | cycles |",
    "|---|---|---:|",
]
for k in kernels:
    acc = "PASS" if k["pass"] else "FAIL"
    cyc = f"{k['cycles']:,}" if k["cycles"] is not None else "-"
    lines.append(f"| {k['name']} | {acc} | {cyc} |")
if not kernels:
    lines.append("| (no kernel results in output) | | |")
md = "\n".join(lines) + "\n"
(out / "summary.md").write_text(md)
print(md, end="")
EOF

echo
echo "saved to $out/: status.json log.txt result.txt summary.json summary.md"
