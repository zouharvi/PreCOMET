#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/PreCOMET


# rsync -azP --filter=":- .gitignore" --exclude .git/ data/csv/ euler:/cluster/work/sachan/vilem/PreCOMET/data/csv/
# rsync -azP euler:/cluster/work/sachan/vilem/PreCOMET/lightning_logs/version_19777971/ lightning_logs/version_19777971/