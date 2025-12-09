#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_proto_files>"
    exit 1
fi

PROTO_PATH="$1"

mkdir -p owocr/py_lens
protoc "--proto_path=$PROTO_PATH" --python_out=owocr/py_lens $PROTO_PATH/*.proto
sed -i '' "s/import \(.*_pb2\) as \(.*\)/from . import \1 as \2/" owocr/py_lens/*