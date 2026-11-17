#!/usr/bin/env bash
# Install the Wavelet Prosody Toolkit (the tool that produced the corpus labels) at the
# pinned commit, outside this package: its source in $WPT_SRC and its dependencies in
# $WPT_LIB. Afterwards set WPT_PATH as printed, for prosody_llms.audio.reconstruct.
#
#   bash scripts/install_toolkit.sh
#
# Defaults: tools/wpt-src and tools/wpt-lib in the repository. Needs git and internet.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=$(python -c "from prosody_llms.audio.reconstruct import TOOLKIT_COMMIT; print(TOOLKIT_COMMIT)")
WPT_SRC=${WPT_SRC:-$ROOT/tools/wpt-src}
WPT_LIB=${WPT_LIB:-$ROOT/tools/wpt-lib}

if [[ ! -d $WPT_SRC/.git ]]; then
  git clone -q https://github.com/asuni/wavelet_prosody_toolkit "$WPT_SRC"
fi
git -C "$WPT_SRC" fetch -q origin
git -C "$WPT_SRC" checkout -q "$COMMIT"
python -m pip install -q --target "$WPT_LIB" \
  "git+https://github.com/asuni/wavelet_prosody_toolkit@$COMMIT" matplotlib
PYTHONPATH=$WPT_SRC:$WPT_LIB MPLBACKEND=Agg \
  python -c "import wavelet_prosody_toolkit.prosody_labeller" && echo "toolkit installed at $COMMIT"
echo "prosody_llms finds it in tools/; with other locations: export WPT_PATH=$WPT_SRC:$WPT_LIB"
