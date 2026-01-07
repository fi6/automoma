#!/bin/bash
# r_h100.sh — Sync local code to cluster (excluding large/output dirs)

# ======================
# Configuration
# ======================
USER="xinhai"
HOST="222.223.127.248"
PORT="18822"
KEY="$HOME/.ssh/id_ed25519"

PROJECT_NAME="automoma"
# Local project directory
LOCAL_DIR="$HOME/projects/$PROJECT_NAME/"    
# Remote destination directory
REMOTE_DIR="/home/xinhai/projects/$PROJECT_NAME"  

# Directories to exclude during rsync (space-separated, relative to LOCAL_DIR)
EXCLUDE_DIRS=(
)

# ======================
# Build rsync exclude flags
# ======================
RSYNC_EXCLUDE_FLAGS=""
for dir in "${EXCLUDE_DIRS[@]}"; do
    RSYNC_EXCLUDE_FLAGS+="--exclude='$dir' "
done

# ======================
# Step 1: Sync code
# ======================
echo "Syncing $LOCAL_DIR → $USER@$HOST:$REMOTE_DIR"
rsync -avz \
      $RSYNC_EXCLUDE_FLAGS \
      -e "ssh -p $PORT -i $KEY" \
      "$LOCAL_DIR" "$USER@$HOST:$REMOTE_DIR"

if [ $? -ne 0 ]; then
    echo "rsync failed. Aborting."
    exit 1
fi

echo "Sync completed."