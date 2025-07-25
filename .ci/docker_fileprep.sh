#!/usr/bin/env bash

current_dir=$(dirname "$0")
cp -r "$current_dir/../llvm-project" "$current_dir"
cp "$current_dir/../requirements.txt" "$current_dir"
