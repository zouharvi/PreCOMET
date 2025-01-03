#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-src

# rsync -azP euler:/cluster/work/sachan/vilem/comet-src/lightning_logs/version_19777971/ lightning_logs/version_19777971/