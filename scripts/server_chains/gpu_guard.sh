#!/bin/bash
# Release a GPU as soon as somebody else starts using it (Daniele, 2026-10-15: "usa la gpu 2 con
# il controllo attivo"). Watches the compute processes of GPU <gpu>; when one belongs to another
# user it stops OUR container <name> -- never anything else -- and exits. It also exits when the
# container ends on its own. It never restarts anything.
#   setsid nohup bash outputs/gpu_guard.sh <gpu> <container name> > outputs/gpu_guard_<gpu>.log 2>&1 < /dev/null &
set -u
GPU=$1
NAME=$2
ME=$(id -u)
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
echo "$(date '+%m-%d %H:%M:%S') guard on GPU $GPU ($UUID) for container $NAME, my uid $ME"
for i in $(seq 1 60); do                      # wait up to 10 min for the container to appear
  docker ps --format '{{.Names}}' | grep -qx "$NAME" && break
  sleep 10
done
while docker ps --format '{{.Names}}' | grep -qx "$NAME"; do
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$uid" ] && [ "$uid" != "$ME" ]; then
      who=$(ps -o user=,cmd= -p "$pid" 2>/dev/null | cut -c1-100)
      echo "$(date '+%m-%d %H:%M:%S') another user's process on GPU $GPU: pid $pid ($who) -> stopping $NAME"
      docker stop "$NAME" >/dev/null
      echo "$(date '+%m-%d %H:%M:%S') GUARD_STOPPED"
      exit 0
    fi
  done
  sleep 15
done
echo "$(date '+%m-%d %H:%M:%S') $NAME is no longer running, guard exits"
