#!/usr/bin/env bash
cd ./deeplearning_data
find . -type d -empty -not -path "./.git/*" -exec touch {}/.gitkeep \;
cd -
