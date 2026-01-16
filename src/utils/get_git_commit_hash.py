import subprocess
from typing import Optional


def get_git_commit_hash() -> Optional[str]:
    """Return the current Git HEAD commit hash or None if unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return None
