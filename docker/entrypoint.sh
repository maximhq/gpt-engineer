#!/usr/bin/env bash
# -*- coding: utf-8 -*-

project_dir="/project"

# Set default environment
ENVIRONMENT=${ENVIRONMENT:-development}

# Check if we should run in CLI mode
if [ "$1" = "--cli" ]; then
    shift  # Remove the --cli argument
    # Run the gpt engineer script in multi-turn mode
    gpt-engineer -mt $project_dir "$@"

    # Patch the permissions of the generated files to be owned by nobody except prompt file
    find "$project_dir" -mindepth 1 -maxdepth 1 ! -path "$project_dir/prompt" -exec chown -R nobody:nogroup {} + -exec chmod -R 777 {} +
else
    # Run the web UI by default
    echo "🚀 Starting GPT Engineer Web UI..."
    echo "📱 Web UI will be available at: http://0.0.0.0:5001"
    echo "   Environment: $ENVIRONMENT"

    if [ "$ENVIRONMENT" = "production" ]; then
        echo "   Running in production mode with Gunicorn"
        # Run with Gunicorn for production
        gunicorn --config gunicorn.conf.py "gpt_engineer.applications.web.app:app"
    else
        echo "   Running in development mode with Flask dev server"
        # Run the web application in development mode
        python -m gpt_engineer.applications.web --host 0.0.0.0 --port 5001
    fi
fi