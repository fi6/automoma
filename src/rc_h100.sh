#!/bin/bash
# rsync_and_connect_h100.sh — Sync local code to cluster (excluding large/output dirs), then connect interactively

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
REMOTE_DIR="$HOME/projects/$PROJECT_NAME"  

# Slurm interactive job settings
PARTITION="h100"
GPUS=1
CPUS=20
TIME="3-00:00:00"
COMMENT="interactive_debug"

# Directories to exclude during rsync (space-separated, relative to LOCAL_DIR)
EXCLUDE_DIRS=(
    "data"
    "data_ing"
    "data_compressed"
    "*.tar.*"
    "assets"
    "outputs"
    "checkpoints"
    "logs"
    ".git"
    "__pycache__"
    "*.pyc"
    "*.log"
    "viz_results"
)

# ======================
# Build rsync exclude flags
# ======================
RSYNC_EXCLUDE_ARGS=()
for dir in "${EXCLUDE_DIRS[@]}"; do
    RSYNC_EXCLUDE_ARGS+=("--exclude=$dir")
done

# ======================
# Step 1: Sync code
# ======================
echo "Syncing $LOCAL_DIR → $USER@$HOST:$REMOTE_DIR"
rsync -avz \
      "${RSYNC_EXCLUDE_ARGS[@]}" \
      -e "ssh -p $PORT -i $KEY" \
      "$LOCAL_DIR" "$USER@$HOST:$REMOTE_DIR"

if [ $? -ne 0 ]; then
    echo "rsync failed. Aborting."
    exit 1
fi

echo "Sync completed."

# ======================
# Step 2: Connect and start interactive GPU session
# ======================
echo "Connecting to H100 cluster and launching interactive job..."
ssh -t -p "$PORT" -i "$KEY" "$USER@$HOST" \
    "cd $REMOTE_DIR && \
     bash -l -c 'module load slurm && bash'"