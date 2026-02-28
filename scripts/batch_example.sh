#!/usr/bin/env bash
set -euo pipefail

# Example: analyze all .tif images in ./images and write results to ./batch_out
semquant batch ./images --pattern "*.tif" --mode pores --out ./batch_out --pixel-size 0.01
