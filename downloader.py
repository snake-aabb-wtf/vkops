# -*- coding: utf-8 -*-
"""downloader.py — run on the TARGET machine: parallel model download where each
chunk is fetched by a curl subprocess (curl -L preserves the Range header across
hf-mirror redirects; python urllib does not). Threads manage chunks; reassemble;
verify sha256 against the Hub's LFS oids."""
import hashlib
import os
import queue
import subprocess
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated/resolve/main/"
OUT = r"C:\vkops\qwen3-4b"
K = 16
CHUNK = 128 * 1024 * 1024

SHARDS = {
    "model-00001-of-00002.safetensors":
        "c6bbaf4d73c46bd9c4ef1a32cd65e6524fe9e71aa28ca63865c1eb6a7ad51a98",
    "model-00002-of-00002.safetensors":
        "b248ed21539e354dedbab9d73f06b0b2f633396664121fd0b3e1c17bc560840f",
}
SMALL = ["config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"]


def fetch_range(url, start, end, dest):
    """Resumable range fetch: keeps whatever bytes are already in `dest`,
    repeatedly downloading only the missing remainder (append mode)."""
    expected = end - start + 1
    tmp = dest + ".tmp"
    for attempt in range(20):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have > expected:
            raise RuntimeError(f"{dest}: oversized {have}>{expected}")
        if have == expected:
            return True
        cmd = ["curl.exe", "-sL", "--max-time", "1800",
               "-r", f"{start + have}-{end}", "-o", tmp, url]
        r = subprocess.run(cmd, capture_output=True)
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if got:
            with open(tmp, "rb") as a, open(dest, "ab") as b:
                b.write(a.read(got))
            os.remove(tmp)
            have = os.path.getsize(dest)
            if have == expected:
                return True
            print(f"  chunk resume: {have}/{expected} (attempt {attempt+1})", flush=True)
        else:
            print(f"  chunk stall (attempt {attempt+1}, rc={r.returncode})", flush=True)
            time.sleep(2 + attempt)
    return os.path.exists(dest) and os.path.getsize(dest) == expected


def download_shard(name, sha):
    url = BASE + name
    dst = f"{OUT}\\{name}"
    size = None
    for _ in range(5):
        r = subprocess.run(["curl.exe", "-sIL", url], capture_output=True, text=True)
        for line in (r.stdout or "").splitlines():
            if line.lower().startswith("content-length:"):
                size = int(line.split(":")[1].strip())
        if size:
            break
    if not size:
        raise RuntimeError(f"{name}: HEAD failed")
    nchunks = (size + CHUNK - 1) // CHUNK
    print(f"[{name}] {size/1e9:.2f} GB -> {nchunks} chunks x {K} workers", flush=True)

    q = queue.Queue()
    for i in range(nchunks):
        q.put(i)
    ok, lock = {}, threading.Lock()

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            start = i * CHUNK
            end = min(size, start + CHUNK) - 1
            if fetch_range(url, start, end, f"{OUT}\\{name}.part{i:05d}"):
                with lock:
                    ok[i] = True
                    print(f"  chunk {len(ok)}/{nchunks}", flush=True)

    ths = [threading.Thread(target=worker) for _ in range(K)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if len(ok) != nchunks:
        raise RuntimeError(f"{name}: chunks missing ({len(ok)}/{nchunks})")

    h = hashlib.sha256()
    with open(dst, "wb") as out:
        for i in range(nchunks):
            p = f"{OUT}\\{name}.part{i:05d}"
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 22), b""):
                    h.update(blk)
                    out.write(blk)
            os.remove(p)
    got = h.hexdigest()
    if got != sha:
        raise RuntimeError(f"{name}: sha256 mismatch got={got}")
    print(f"[{name}] sha256 OK", flush=True)


def fetch_small(name):
    r = subprocess.run(["curl.exe", "-sL", "--max-time", "300",
                        "-o", f"{OUT}\\{name}", BASE + name])
    if r.returncode != 0 or not os.path.exists(f"{OUT}\\{name}"):
        raise RuntimeError(f"small file failed: {name}")
    print(f"small: {name}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for f in SMALL:
        fetch_small(f)
    for name, sha in SHARDS.items():
        if os.path.exists(f"{OUT}\\{name}"):
            print(f"[{name}] already present, skip", flush=True)
            continue
        download_shard(name, sha)
    print(f"ALL DOWNLOADS COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
