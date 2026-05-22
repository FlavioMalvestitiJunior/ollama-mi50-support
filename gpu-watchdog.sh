#!/bin/bash
# GPU thermal watchdog for AMD MI50 (gfx906/Vega20) on Proxmox VM
# Polls every 1s — warns at WARN_TEMP, kills Ollama at CRITICAL_TEMP,
# restarts Ollama automatically when GPU cools below RESUME_TEMP

CRITICAL_TEMP=90
WARN_TEMP=80
RESUME_TEMP=45
POLL_INTERVAL=1

GPU_TEMP_FILE=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)

if [ -z "$GPU_TEMP_FILE" ]; then
    echo "ERROR: No GPU temperature sensor found" >&2
    exit 1
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [gpu-watchdog] $*"; }

log "Started. Sensor: $GPU_TEMP_FILE  Warn: ${WARN_TEMP}°C  Critical: ${CRITICAL_TEMP}°C  Resume: ${RESUME_TEMP}°C  Poll: ${POLL_INTERVAL}s"

triggered=false
warned=false
last_warn_temp=0

while true; do
    raw=$(cat "$GPU_TEMP_FILE" 2>/dev/null)
    [ -z "$raw" ] && sleep "$POLL_INTERVAL" && continue
    temp=$(( raw / 1000 ))

    if [ "$temp" -ge "$CRITICAL_TEMP" ]; then
        if [ "$triggered" = false ]; then
            log "CRITICAL: GPU at ${temp}°C — parando ollama.service"
            systemctl stop ollama
            triggered=true
            warned=false
        fi
    elif [ "$temp" -ge "$WARN_TEMP" ]; then
        # Log warning a cada 2°C de subida para não spammar
        if [ "$warned" = false ] || [ "$temp" -ge $(( last_warn_temp + 2 )) ]; then
            log "AVISO: GPU a ${temp}°C (crítico: ${CRITICAL_TEMP}°C)"
            warned=true
            last_warn_temp=$temp
        fi
        triggered=false
    else
        if [ "$triggered" = true ] && [ "$temp" -le "$RESUME_TEMP" ]; then
            log "GPU resfriou para ${temp}°C — reiniciando ollama.service"
            systemctl start ollama
            triggered=false
            warned=false
            last_warn_temp=0
        fi
        if [ "$triggered" = false ]; then
            warned=false
            last_warn_temp=0
        fi
    fi

    sleep "$POLL_INTERVAL"
done
