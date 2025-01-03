#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-src

# scp -r euler:/cluster/work/sachan/vilem/comet-src/lightning_logs/version_19777784/ lightning_logs/version_19777784/