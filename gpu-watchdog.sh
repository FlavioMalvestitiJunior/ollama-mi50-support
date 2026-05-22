#!/bin/bash
# GPU thermal watchdog para AMD MI50 (gfx906/Vega20) em Proxmox VM
#
# Limites térmicos do MI50:
#   94-95°C → GPU derruba a si mesma e leva o host Proxmox junto
#
# Escalonamento:
#   >= WARN_TEMP   (80°C) : loga aviso a cada +2°C
#   >= SIGTERM_TEMP(85°C) : SIGTERM em todos os processos GPU;
#                           Ollama via systemctl stop (graceful)
#   >= SIGKILL_TEMP(90°C) : SIGKILL em qualquer processo GPU sobrevivente;
#                           Ollama via systemctl kill --signal=SIGKILL
#   <= RESUME_TEMP (45°C) : reinicia o Ollama automaticamente se foi parado
#
# Requer: psmisc (fuser) — apt install psmisc

WARN_TEMP=80
SIGTERM_TEMP=85
SIGKILL_TEMP=90
RESUME_TEMP=45
POLL_INTERVAL=1

GPU_DEVICES="/dev/dri/renderD128 /dev/kfd"

GPU_TEMP_FILE=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
if [ -z "$GPU_TEMP_FILE" ]; then
    echo "ERROR: No GPU temperature sensor found" >&2
    exit 1
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [gpu-watchdog] $*"; }
log "Started. Sensor: $GPU_TEMP_FILE  Warn: ${WARN_TEMP}°C  SIGTERM: ${SIGTERM_TEMP}°C  SIGKILL: ${SIGKILL_TEMP}°C  Resume: ${RESUME_TEMP}°C  Poll: ${POLL_INTERVAL}s"

sigterm_sent=false
sigkill_sent=false
ollama_was_stopped=false
warned=false
last_warn_temp=0

# Envia $1 (TERM ou KILL) a todos os processos com fd aberto na GPU.
# Para Ollama usa systemd para TERM e systemctl kill para KILL.
signal_gpu_processes() {
    local sig="$1"   # TERM ou KILL
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
            if [ "$sig" = "TERM" ]; then
                if [ "$ollama_was_stopped" = false ] && systemctl is-active ollama >/dev/null 2>&1; then
                    log "  ollama (PID $pid: $name) → systemctl stop (SIGTERM graceful)"
                    systemctl stop ollama
                    ollama_was_stopped=true
                fi
            else
                # SIGKILL: força se ainda estiver rodando
                if systemctl is-active ollama >/dev/null 2>&1; then
                    log "  ollama ainda ativo → systemctl kill --signal=SIGKILL"
                    systemctl kill --signal=SIGKILL ollama
                fi
            fi
        else
            log "  PID $pid ($name) → SIG${sig}"
            kill -"$sig" "$pid" 2>/dev/null
        fi
    done
}

while true; do
    raw=$(cat "$GPU_TEMP_FILE" 2>/dev/null)
    [ -z "$raw" ] && sleep "$POLL_INTERVAL" && continue
    temp=$(( raw / 1000 ))

    if [ "$temp" -ge "$SIGKILL_TEMP" ]; then
        # Nível 2 — SIGKILL: o que sobrou morre agora
        if [ "$sigterm_sent" = false ]; then
            # Pulou direto para 90°C sem passar pelo 85°C (ex: subida abrupta)
            log "SIGTERM: GPU a ${temp}°C (salto direto) — SIGTERM em processos GPU"
            signal_gpu_processes TERM
            sigterm_sent=true
        fi
        if [ "$sigkill_sent" = false ]; then
            log "SIGKILL: GPU a ${temp}°C — SIGKILL em processos GPU sobreviventes"
            signal_gpu_processes KILL
            sigkill_sent=true
        fi

    elif [ "$temp" -ge "$SIGTERM_TEMP" ]; then
        # Nível 1 — SIGTERM: dá chance de encerrar graciosamente
        if [ "$sigterm_sent" = false ]; then
            log "SIGTERM: GPU a ${temp}°C — SIGTERM em processos GPU"
            signal_gpu_processes TERM
            sigterm_sent=true
        fi

    elif [ "$temp" -ge "$WARN_TEMP" ]; then
        # Aviso — não interfere em estado de cooldown
        if [ "$warned" = false ] || [ "$temp" -ge $(( last_warn_temp + 2 )) ]; then
            log "AVISO: GPU a ${temp}°C  (SIGTERM: ${SIGTERM_TEMP}°C | SIGKILL: ${SIGKILL_TEMP}°C)"
            warned=true
            last_warn_temp=$temp
        fi

    elif [ "$sigterm_sent" = true ] || [ "$sigkill_sent" = true ]; then
        # Cooldown: aguarda atingir RESUME_TEMP para religar o Ollama
        if [ "$temp" -le "$RESUME_TEMP" ]; then
            if [ "$ollama_was_stopped" = true ]; then
                log "GPU resfriou para ${temp}°C — reiniciando ollama.service"
                systemctl start ollama
                ollama_was_stopped=false
            fi
            sigterm_sent=false
            sigkill_sent=false
            warned=false
            last_warn_temp=0
        fi

    else
        warned=false
        last_warn_temp=0
    fi

    sleep "$POLL_INTERVAL"
done
