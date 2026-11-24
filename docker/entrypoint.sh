#!/bin/bash
# Entrypoint for the SliceLens container.
#
# AIRLab launches containers with `-u <uid>:<gid>` (the host user). That uid has
# no entry in the image's /etc/passwd, which breaks libraries that call
# getpwuid() (and leaves $HOME unset). We register the uid and ensure a writable
# HOME, then exec the requested command.
set -e

USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
export HOME="${HOME:-/home/exp}"

# Add a passwd entry for this uid if missing (and /etc/passwd is writable).
if ! id -nu "${USER_ID}" >/dev/null 2>&1; then
    if [ -w /etc/passwd ]; then
        echo "exp:x:${USER_ID}:${GROUP_ID}:exp:${HOME}:/bin/bash" >> /etc/passwd
    fi
fi

mkdir -p "${HOME}" 2>/dev/null || true

exec "$@"
