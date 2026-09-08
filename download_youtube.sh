#!/usr/bin/env bash
# Download a YouTube video to $YOUTUBE_DIR/<name>.mp4 via yt-dlp (bash equiv of download_youtube.bat).
#   ./download_youtube.sh <url> [name] [max_height]
source "$(dirname "$0")/_env.sh"

URL="${1:-}"
[ -z "$URL" ] && { echo "Usage: ./download_youtube.sh <url> [name] [max_height]"; exit 1; }
NAME="${2:-}"
MAXH="${3:-}"

mkdir -p "$YOUTUBE_DIR"
if [ -z "$NAME" ]; then OUT_TMPL="$YOUTUBE_DIR/%(id)s.%(ext)s"; else OUT_TMPL="$YOUTUBE_DIR/$NAME.%(ext)s"; fi
if [ -z "$MAXH" ]; then
    FMT="bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
else
    FMT="bestvideo[ext=mp4][height<=$MAXH]+bestaudio[ext=m4a]/best[ext=mp4][height<=$MAXH]/best"
fi

DL="yt-dlp"; command -v yt-dlp >/dev/null 2>&1 || DL="uvx yt-dlp"
echo "Downloading $URL -> $YOUTUBE_DIR/$NAME.mp4"
$DL -f "$FMT" --merge-output-format mp4 -o "$OUT_TMPL" "$URL"
echo "Done. Saved under $YOUTUBE_DIR"
