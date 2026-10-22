#!/bin/sh
# Assemble the GitHub Pages site into _site/: the page plus the images it
# shares with the README, copied rather than duplicated in the repository.
# The pages workflow runs this; run it locally to preview:
#   site/build.sh && python -m http.server -d _site
set -eu
cd "$(dirname "$0")/.."
rm -rf _site
mkdir -p _site/assets
cp site/index.html _site/
cp src/rcta/gui/icons/rcta.svg \
   docs/screenshot-walk.png docs/screenshot-flight-fit.png docs/social-preview.png \
   _site/assets/
touch _site/.nojekyll
echo "built _site/ ($(du -sh _site | cut -f1))"
