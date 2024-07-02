#!/bin/bash

OWNER_UID=$(stat -c '%u' "${HOME_PATH}")
OWNER_GID=$(stat -c '%g' "${HOME_PATH}")

sed -i "s/:1000:1000:/:${OWNER_UID}:${OWNER_GID}:/" /etc/passwd
sed -i "s/:1000:/:${OWNER_GID}:/" /etc/group

sudo -u user -g user -H env "PATH=$PATH" "$@"
