# -*- coding: utf-8 -*-
"""deploy.py — push this repo's sources to the target machine via SFTP.

Credentials come from the environment (never hardcoded):
  VKOPS_SSH_HOST (default 100.105.188.76)
  VKOPS_SSH_USER (default administrator)
  VKOPS_SSH_PASS (required)

Usage:  python deploy.py [remote_dir]      # default remote dir: C:/vkops
"""
import os
import sys

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = os.environ.get("VKOPS_SSH_HOST", "100.105.188.76")
USER = os.environ.get("VKOPS_SSH_USER", "administrator")
PWD = os.environ.get("VKOPS_SSH_PASS", "")
REMOTE = (sys.argv[1] if len(sys.argv) > 1 else "C:/vkops").rstrip("/")

LOCAL = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".git", "ref", "__pycache__"}
SKIP_EXT = {".safetensors", ".spv"}


def ensure_dir(sftp, rdir):
    cur = ""
    for p in rdir.replace("\\", "/").split("/"):
        if not p:
            continue
        cur = (cur.rstrip("/") + "/" + p) if cur else (p if p.endswith(":") else p)
        if p.endswith(":"):
            cur = p + "/"
            continue
        try:
            sftp.stat(cur.rstrip("/"))
        except FileNotFoundError:
            sftp.mkdir(cur.rstrip("/"))


def main():
    if not PWD:
        print("error: set VKOPS_SSH_PASS first")
        return 1
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PWD, allow_agent=False, look_for_keys=False, timeout=15)
    sftp = c.open_sftp()
    ensure_dir(sftp, REMOTE)
    n = 0
    for root, dirs, files in os.walk(LOCAL):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, LOCAL).replace("\\", "/")
        rdir = REMOTE + ("/" + rel if rel != "." else "")
        ensure_dir(sftp, rdir)
        for f in files:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            sftp.put(os.path.join(root, f), rdir.rstrip("/") + "/" + f)
            n += 1
    print(f"deployed {n} files -> {USER}@{HOST}:{REMOTE}")
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
