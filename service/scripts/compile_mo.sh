#!/bin/sh

set -e
# set -x  # show every command

LOCALES_DIR="./locales"

if [ ! -d "$LOCALES_DIR" ]; then
    echo "Locales directory '$LOCALES_DIR' not found!"
    exit 1
fi

echo "Compiling .po files in $LOCALES_DIR ..."

find "$LOCALES_DIR" -type f -name "*.po" | while read po; do
    mo="${po%.po}.mo"
    echo "Compiling $po -> $mo"
    msgfmt "$po" -o "$mo"
done

echo "Done compiling all .po files."
