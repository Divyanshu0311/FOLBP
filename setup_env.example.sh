#!/usr/bin/env bash
# Template for setup_env.sh — copy it, fill in your key, and source it.
#
#   cp setup_env.example.sh setup_env.sh
#   # edit setup_env.sh, put your real key in
#   source setup_env.sh
#
# setup_env.sh is listed in .gitignore and must never be committed.
# This template is committed and must never contain a real key.
#
# Every entry point (FOLBP, PEFA, CRMS, DRMS, PEFA_wo_history, mcts, the
# bench sweep) reads the key from the environment; none of them carry a
# default. Passing --api_key on the command line still overrides this.

# Gemini / Google AI Studio key — https://aistudio.google.com/apikey
export GEMINI_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"
export GOOGLE_API_KEY="$GEMINI_API_KEY"

# Only needed if you run the original OpenAI-backed baselines.
# export OPENAI_API_KEY="PASTE_YOUR_OPENAI_API_KEY_HERE"

# Absolute path to this checkout, used by the OmniGibson / ROS 2 side.
export COHERENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"

if [ "$GEMINI_API_KEY" = "PASTE_YOUR_GEMINI_API_KEY_HERE" ]; then
    echo "setup_env.sh: GEMINI_API_KEY is still the placeholder — edit this file." >&2
else
    echo "setup_env.sh: GEMINI_API_KEY, GOOGLE_API_KEY and COHERENT_PATH exported."
fi
