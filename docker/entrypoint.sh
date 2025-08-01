#!/usr/bin/env bash
# -*- coding: utf-8 -*-

project_dir="/project"

# Run the gpt engineer script in multi-turn mode by default
# If additional arguments are provided, they will be appended after --multi-turn
gpt-engineer -mt $project_dir "$@"

# Patch the permissions of the generated files to be owned by nobody except prompt file
find "$project_dir" -mindepth 1 -maxdepth 1 ! -path "$project_dir/prompt" -exec chown -R nobody:nogroup {} + -exec chmod -R 777 {} +
