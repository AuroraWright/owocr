#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_proto_files>"
    exit 1
fi

PROTO_PATH="$1"

mkdir -p owocr/py_lens
protoc "--proto_path=$PROTO_PATH" --python_out=owocr/py_lens $PROTO_PATH/*.proto
sed -i '' "s/import lens_overlay/import owocr.py_lens.lens_overlay/" py_lens/*