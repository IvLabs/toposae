#!/bin/bash
# Helper script for project management

show_status() {
    echo "=== Project Status ==="
    git status
    echo ""
    echo "=== Recent Commits ==="
    git log --oneline -5
}

start_experiment() {
    local name=$1
    if [ -z "$name" ]; then
        echo "Usage: $0 start-exp <experiment_name>"
        exit 1
    fi
    git checkout -b experiment/$name
    echo "Created branch: experiment/$name"
    echo "Remember to update PROGRESS.md with experiment details!"
}

commit_work() {
    local message=$1
    if [ -z "$message" ]; then
        echo "Usage: $0 commit <message>"
        exit 1
    fi
    git add .
    git commit -m "$message"
    echo "Committed: $message"
}

show_progress() {
    echo "=== Latest Progress ==="
    tail -20 PROGRESS.md
}

case "$1" in
    status)
        show_status
        ;;
    start-exp)
        start_experiment "$2"
        ;;
    commit)
        commit_work "$2"
        ;;
    progress)
        show_progress
        ;;
    *)
        echo "Usage: $0 {status|start-exp|commit|progress}"
        echo ""
        echo "Commands:"
        echo "  status     - Show git status and recent commits"
        echo "  start-exp  - Start new experiment branch"
        echo "  commit     - Commit all changes with message"
        echo "  progress   - Show latest progress"
        exit 1
        ;;
esac
