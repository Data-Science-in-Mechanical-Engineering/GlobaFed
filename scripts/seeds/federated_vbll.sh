#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
python -m fedbo.main -cn=federated_vbll --multirun +hydra=seeds "$@"
