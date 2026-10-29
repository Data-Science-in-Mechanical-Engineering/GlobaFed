#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
python -m fedbo.main -cn=local_mlp "$@"
