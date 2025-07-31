#!/bin/bash

# This script should run in tmux

# Default values
loops=10
port=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --loops)
      loops="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done


tmux new-window -n "t_all"
tmux split-window -h
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 2
tmux split-window -v

tmux send-keys -t "t_all.0" "script -f -c \"python3 prefill_test.py --loops $loops --port $port\" run_0_prefill.log" C-m
tmux send-keys -t "t_all.1" "script -f -c \"python3 prefill_test.py --loops $loops --port $port\" run_1_prefill.log" C-m
tmux send-keys -t "t_all.2" "script -f -c \"python3 prefill_test.py --loops $loops --port $port\" run_2_prefill.log" C-m
tmux send-keys -t "t_all.3" "script -f -c \"python3 prefill_test.py --loops $loops --port $port\" run_3_prefill.log" C-m