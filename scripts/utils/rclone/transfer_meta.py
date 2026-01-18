import argparse
import subprocess
import sys
import shlex
def run_rclone_sync(rclone, src_path, dst_path, dry_run=False):
    """
    Transfer meta data using rclone.
    Logic:
    1. Recursively scan the src_path.
    2. Match all folders named 'meta' and their sub-contents.
    3. Ignore all other files (data, images, videos, etc.).
    4. Copy to dst_path while preserving the directory structure.
    """
    
    # Rclone command construction
    # copy: copy files
    # --include "**/meta/**": core logic.
    #    "**" matches any intermediate directory
    #    "/meta/" matches folders named meta
    #    "/**" matches all content under the meta folder
    # -P: show progress
    # --transfers: number of concurrent transfers (optional, rclone default is usually sufficient)
    
    cmd = [
        rclone, "copy", src_path, dst_path,
        "--include", "**/meta/**",
        "-P",  # Show real-time progress
        "--stats-one-line" # Concise progress output
    ]

    if dry_run:
        cmd.append("--dry-run")
        print("\n[Dry Run] Simulating execution, no actual modifications will be made.\n")

    # Print the generated command for debugging
    print(f"Executing command: {' '.join(shlex.quote(arg) for arg in cmd)}")
    print("-" * 40)

    try:
        # Call system rclone
        result = subprocess.run(cmd, check=True, text=True)
        print("-" * 40)
        print("✅ Operation completed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Rclone execution failed (Exit code: {e.returncode})")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("❌ rclone command not found. Please ensure rclone is installed and added to the PATH environment variable.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively transfer Lerobot Dataset meta folders (supports bidirectional Local/Remote transfer)"
    )
    parser.add_argument(
        "rclone",
        type=str,
        default="rclone",
        help="Path to rclone executable (e.g. /home/xinhai/env/rclone-v1.72.1-linux-amd64/rclone or just rclone if it's in PATH)"
    )
    parser.add_argument(
        "src_path", 
        help="Source path (e.g.: /home/user/data or 123pan:/data)"
    )
    parser.add_argument(
        "dst_path", 
        help="Destination path (e.g.: 123pan:/data_meta or /home/user/backup)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Dry run mode, only list files to be transferred without actual copying"
    )

    args = parser.parse_args()

    # Simple path cleanup (prevent users from entering paths with extra spaces)
    rclone = args.rclone.strip()
    src = args.src_path.strip()
    dst = args.dst_path.strip()

    print(f"Source: {src}")
    print(f"Dest  : {dst}")
    
    run_rclone_sync(rclone, src, dst, args.dry_run)