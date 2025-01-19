#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/PreCOMET

# scp data/csv/* euler:/cluster/work/sachan/vilem/PreCOMET/data/csv/
# scp -r euler:/cluster/work/sachan/vilem/PreCOMET/lightning_logs/version_{0,1,2} lightning_logs/