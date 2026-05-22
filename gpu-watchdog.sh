#!/bin/bash
# GPU thermal watchdog for AMD MI50 (gfx906/Vega20) on Proxmox VM
#
# Comportamento:
#   >= WARN_TEMP (80°C)     : loga aviso a cada +2°C
#   >= CRITICAL_TEMP (90°C) : detecta todo processo com fd aberto nos device
#                             nodes da GPU (fuser), mata cada um; se o Ollama
#                             estava entre eles, reinicia automaticamente quando
#                             a GPU esfriar abaixo de RESUME_TEMP (45°C)
#
# Requer: psmisc (fuser) — instale com: apt install psmisc

CRITICAL_TEMP=90
WARN_TEMP=80
RESUME_TEMP=45
POLL_INTERVAL=1

# Device nodes — qualquer processo com fd aberto aqui está usando a GPU
GPU_DEVICES="/dev/dri/renderD128 /dev/kfd"

GPU_TEMP_FILE=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
if [ -z "$GPU_TEMP_FILE" ]; then
    echo "ERROR: No GPU temperature sensor found" >&2
    exit 1
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [gpu-watchdog] $*"; }
log "Started. Sensor: $GPU_TEMP_FILE  Warn: ${WARN_TEMP}°C  Critical: ${CRITICAL_TEMP}°C  Resume: ${RESUME_TEMP}°C  Poll: ${POLL_INTERVAL}s"

triggered=false
ollama_was_stopped=false
warned=false
last_warn_temp=0

kill_gpu_processes() {
    # fuser precisa de root para ver processos de outros usuários (ex: ollama)
    local gpu_pids
    gpu_pids=$(fuser $GPU_DEVICES 2>/dev/null | tr ' ' '\n' | grep -v '^$' | sort -u)

    if [ -z "$gpu_pids" ]; then
        log "  Nenhum processo com GPU fd encontrado"
        return
    fi

    for pid in $gpu_pids; do
        [ -d "/proc/$pid" ] || continue
        local name cmdline
        name=$(ps -p "$pid" -o comm= 2>/dev/null || echo "?")
        cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)

        if echo "$cmdline" | grep -q 'ollama'; then
            # Para via systemd — evita que o Restart=always relance imediatamente
            if [ "$ollama_was_stopped" = false ] && systemctl is-active ollama >/dev/null 2>&1; then
                log "  ollama (PID $pid: $name) → systemctl stop ollama"
                systemctl stop ollama
                ollama_was_stopped=true
            fi
        else
            # Processo GPU desconhecido: SIGTERM, depois SIGKILL após 3s
            log "  processo GPU (PID $pid: $name) → SIGTERM"
            kill -TERM "$pid" 2>/dev/null
            sleep 3
            if kill -0 "$pid" 2>/dev/null; then
                log "  processo GPU (PID $pid: $name) não terminou → SIGKILL"
                kill -KILL "$pid" 2>/dev/null
            fi
        fi
    done
}

while true; do
    raw=$(cat "$GPU_TEMP_FILE" 2>/dev/null)
    [ -z "$raw" ] && sleep "$POLL_INTERVAL" && continue
    temp=$(( raw / 1000 ))

    if [ "$temp" -ge "$CRITICAL_TEMP" ]; then
        if [ "$triggered" = false ]; then
            log "CRITICAL: GPU a ${temp}°C — matando processos GPU"
            kill_gpu_processes
            triggered=true
            warned=false
        fi

    elif [ "$temp" -ge "$WARN_TEMP" ]; then
        # Não altera triggered — pode estar em cooldown após critical
        if [ "$warned" = false ] || [ "$temp" -ge $(( last_warn_temp + 2 )) ]; then
            log "AVISO: GPU a ${temp}°C (crítico: ${CRITICAL_TEMP}°C)"
            warned=true
            last_warn_temp=$temp
        fi

    elif [ "$triggered" = true ]; then
        # Cooldown: desceu do crítico, aguardando chegar em RESUME_TEMP
        if [ "$temp" -le "$RESUME_TEMP" ]; then
            if [ "$ollama_was_stopped" = true ]; then
                log "GPU resfriou para ${temp}°C — reiniciando ollama.service"
                systemctl start ollama
                ollama_was_stopped=false
            fi
            triggered=false
            warned=false
            last_warn_temp=0
        fi

    else
        # Normal: limpa estado de aviso
        warned=false
        last_warn_temp=0
    fi

    sleep "$POLL_INTERVAL"
done
