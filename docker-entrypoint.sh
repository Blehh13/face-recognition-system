#!/bin/sh
# Container entrypoint.
#
# The demo wipe lives here rather than in the application because gunicorn
# workers initialise lazily: clearing the database on first use meant the
# second worker erased everything the first had enrolled, the moment it
# handled its first request. Here it runs exactly once, before any worker
# exists.
set -e

if [ "$FACEREC_DEMO" = "1" ] && [ -n "$FACEREC_DB" ]; then
    rm -f "$FACEREC_DB" "$FACEREC_DB-wal" "$FACEREC_DB-shm"
    echo "demo mode: starting with an empty database"
fi

exec "$@"
