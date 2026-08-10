#!/bin/sh
set -eu

data_root="${DATA_ROOT:-/data}"
mkdir -p "$data_root/uploads" "$data_root/reports" "$data_root/instance"
chown -R regressai:regressai "$data_root"

exec gosu regressai "$@"
