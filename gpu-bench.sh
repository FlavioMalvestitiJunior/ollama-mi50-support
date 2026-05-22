#!/bin/bash
# gpu-bench.sh — Ollama benchmark with thermal cooldown between runs
# Usage: gpu-bench.sh [model] [prompt] [runs]
# Waits for GPU to drop below COOL_TEMP before each run after the first.

COOL_TEMP=45
POLL_INTERVAL=30
MODEL="${1:-gemma4:31b}"
PROMPT="${2:-What is photosynthesis? Answer in 60 words.}"
RUNS="${3:-3}"

GPU_TEMP_FILE=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)

get_temp() { echo $(( $(cat "$GPU_TEMP_FILE" 2>/dev/null || echo 0) / 1000 )); }

cooldown() {
    local temp
    temp=$(get_temp)
    [ "$temp" -le "$COOL_TEMP" ] && return
    echo "  GPU at ${temp}°C — esperando esfriar até ${COOL_TEMP}°C (intervalos de ${POLL_INTERVAL}s)..."
    while [ "$temp" -gt "$COOL_TEMP" ]; do
        sleep "$POLL_INTERVAL"
        temp=$(get_temp)
        echo "  GPU: ${temp}°C"
    done
    echo "  GPU esfriou para ${temp}°C — retomando"
}

echo "=== GPU Benchmark: $MODEL  ($RUNS runs, cooldown <${COOL_TEMP}°C) ==="
echo "    GPU temp inicial: $(get_temp)°C"

for i in $(seq 1 "$RUNS"); do
    [ "$i" -gt 1 ] && cooldown
    echo "--- Run $i/$RUNS  (GPU: $(get_temp)°C) ---"
    ollama run "$MODEL" "$PROMPT" --verbose 2>&1 | grep -E 'eval rate'
done

echo "=== Concluído. GPU: $(get_temp)°C ==="
