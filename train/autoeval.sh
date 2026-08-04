#!/usr/bin/env bash
# Waits for fine-tune checkpoint milestones and runs the paired seat
# eval (tuned vs pretrained baseline) at each. Appends to autoeval.log.
set -u
cd "$(dirname "$0")/.."
BASE=vendor/upstream/resources/nmmo3/nmmo3_weights.bin
CKPT=train/ckpt_v1
LOG=train/autoeval.log
PY=train/.venv/bin/python

for M in 000800 001600 003200 006400 009100; do
    F=$CKPT/weights_$M.bin
    echo "[autoeval] waiting for $F" >> $LOG
    while [ ! -f "$F" ]; do
        # stop waiting if the trainer died and no newer ckpt will come
        if ! pgrep -f "train/ppo.py" > /dev/null; then
            LATEST=$(ls -t $CKPT/weights_*.bin 2>/dev/null | head -1)
            if [ -z "$LATEST" ]; then
                echo "[autoeval] trainer gone, no checkpoints - abort" >> $LOG
                exit 1
            fi
            if [ "$LATEST" != "$F" ]; then
                echo "[autoeval] trainer gone; final eval on $LATEST" >> $LOG
                F=$LATEST
            fi
            break
        fi
        sleep 60
    done
    echo "[autoeval] === eval $F vs baseline ($(date)) ===" >> $LOG
    $PY train/eval_seats.py "$BASE" "$F" 700 14 5000 >> $LOG 2>&1
    # if the trainer is gone and we just evaluated the last ckpt, stop
    if ! pgrep -f "train/ppo.py" > /dev/null; then
        LATEST=$(ls -t $CKPT/weights_*.bin | head -1)
        [ "$F" = "$LATEST" ] && break
    fi
done
echo "[autoeval] ladder done ($(date))" >> $LOG
