"""
Fetch your Pinterest boards via the Pinterest API v5 and build a training pairs CSV.

Positive pairs : two pins from the SAME board (curated together = compatible)
Negative pairs : one pin from board A + one pin from board B (different taste boards)

Prerequisites
-------------
1. Create a Pinterest app at https://developers.pinterest.com/apps/
   Request scopes: boards:read  pins:read
2. Run the auth step to get an access token:
       venv/bin/python scripts/fetch_pinterest_boards.py \
           --auth \
           --app-id YOUR_APP_ID \
           --app-secret YOUR_APP_SECRET
   Follow the printed instructions to complete the OAuth flow.
3. Pass the token via --access-token or set PINTEREST_TOKEN env var.

Usage
-----
# Step 1: start OAuth flow
    venv/bin/python scripts/fetch_pinterest_boards.py \
        --auth --app-id <ID> --app-secret <SECRET>

# Step 2: exchange the code printed after redirect (the script does this for you)
    venv/bin/python scripts/fetch_pinterest_boards.py \
        --auth --app-id <ID> --app-secret <SECRET> --code <CODE>

# Step 3: download boards and build pairs
    venv/bin/python scripts/fetch_pinterest_boards.py \
        --access-token <TOKEN> \
        --output-dir data/pinterest

# Step 4: train
    venv/bin/python scripts/train_polyvore.py \
        --pairs-csv data/pinterest/pairs.csv \
        --epochs 5
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

PINTEREST_AUTH_URL = "https://www.pinterest.com/oauth/"
PINTEREST_TOKEN_URL = "https://api.pinterest.com/v5/oauth/token"
PINTEREST_API_BASE = "https://api.pinterest.com/v5"
SCOPES = "boards:read,pins:read"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch Pinterest boards and build outfit-compatibility training pairs."
    )

    auth_group = p.add_argument_group("OAuth (run once to get a token)")
    auth_group.add_argument("--auth", action="store_true", help="Start / complete the OAuth flow.")
    auth_group.add_argument("--app-id", help="Pinterest app ID.")
    auth_group.add_argument("--app-secret", help="Pinterest app secret.")
    auth_group.add_argument("--code", help="Authorization code from redirect URL (step 2 of OAuth).")
    auth_group.add_argument(
        "--redirect-uri", default="https://localhost/callback",
        help="Must match the redirect URI registered in your Pinterest app.",
    )

    run_group = p.add_argument_group("Download & pair generation")
    run_group.add_argument("--access-token", help="Pinterest OAuth2 access token.")
    run_group.add_argument(
        "--output-dir", type=Path, default=Path("data/pinterest"),
        help="Root output directory (images go in <output-dir>/images/).",
    )
    run_group.add_argument(
        "--board-limit", type=int, default=30,
        help="Max number of boards to fetch (0 = all).",
    )
    run_group.add_argument(
        "--pins-per-board", type=int, default=100,
        help="Max pins to download per board.",
    )
    run_group.add_argument(
        "--pairs-per-board", type=int, default=30,
        help="Max positive pairs to generate per board.",
    )
    run_group.add_argument(
        "--neg-ratio", type=float, default=1.0,
        help="Negative pairs per positive pair (default 1.0 = balanced).",
    )
    run_group.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def build_auth_url(app_id: str, redirect_uri: str) -> str:
    params = urllib.parse.urlencode({
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
    })
    return f"{PINTEREST_AUTH_URL}?{params}"


def exchange_code_for_token(app_id: str, app_secret: str, code: str, redirect_uri: str) -> dict:
    credentials = base64.b64encode(f"{app_id}:{app_secret}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }).encode()
    req = urllib.request.Request(
        PINTEREST_TOKEN_URL,
        data=body,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Pinterest API helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, token: str, params=None) -> dict:
    url = f"{PINTEREST_API_BASE}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Pinterest API error {exc.code} on {endpoint}: {body}") from exc


def paginate(endpoint: str, token: str, page_size: int, limit: int) -> list:
    results = []
    bookmark = None
    while True:
        params: dict = {"page_size": min(page_size, limit - len(results))}
        if bookmark:
            params["bookmark"] = bookmark
        data = api_get(endpoint, token, params)
        batch = data.get("items", [])
        results.extend(batch)
        bookmark = data.get("bookmark")
        if not batch or not bookmark or len(results) >= limit:
            break
    return results[:limit]


def best_image_url(pin: dict):
    media = pin.get("media") or {}
    images = media.get("images") or {}
    for size in ("600x", "400x300", "236x", "150x150", "original"):
        entry = images.get(size)
        if isinstance(entry, dict) and entry.get("url"):
            return entry["url"]
    # fallback: first available
    for entry in images.values():
        if isinstance(entry, dict) and entry.get("url"):
            return entry["url"]
    return None


def download_image(url: str, dest: Path, retries: int = 2) -> bool:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as exc:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"    warning: {dest.name} — {exc}")
    return False


def safe_slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "board")).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- OAuth flow -------------------------------------------------------
    if args.auth:
        if not args.app_id or not args.app_secret:
            print(
                "Both --app-id and --app-secret are required for --auth.\n"
                "Create an app at https://developers.pinterest.com/apps/",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.code:
            print("Exchanging authorization code for access token …")
            try:
                token_data = exchange_code_for_token(
                    args.app_id, args.app_secret, args.code, args.redirect_uri
                )
            except Exception as exc:
                print(f"Token exchange failed: {exc}", file=sys.stderr)
                sys.exit(1)

            access_token = token_data.get("access_token", "")
            refresh_token = token_data.get("refresh_token", "")
            print("\nSuccess!")
            print(f"\n  Access token : {access_token}")
            if refresh_token:
                print(f"  Refresh token: {refresh_token}")
            print(
                f"\nSave the access token, then run:\n"
                f"  export PINTEREST_TOKEN='{access_token}'\n"
                f"  venv/bin/python scripts/fetch_pinterest_boards.py"
                f" --output-dir data/pinterest"
            )
        else:
            auth_url = build_auth_url(args.app_id, args.redirect_uri)
            print(
                f"\nStep 1 — Open this URL in your browser to authorize:\n\n"
                f"  {auth_url}\n\n"
                f"Step 2 — After redirecting to your redirect URI, copy the `code`"
                f" query parameter from the URL bar.\n\n"
                f"Step 3 — Run:\n"
                f"  venv/bin/python scripts/fetch_pinterest_boards.py \\\n"
                f"    --auth \\\n"
                f"    --app-id {args.app_id} \\\n"
                f"    --app-secret <SECRET> \\\n"
                f"    --code <CODE_FROM_URL>"
            )
        return

    # ---- Download ---------------------------------------------------------
    token = args.access_token or os.environ.get("PINTEREST_TOKEN", "")
    if not token:
        print(
            "No access token found.\n"
            "Set PINTEREST_TOKEN or pass --access-token.\n"
            "Run with --auth --app-id <ID> --app-secret <SECRET> to start the OAuth flow.",
            file=sys.stderr,
        )
        sys.exit(1)

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = args.output_dir / "pairs.csv"

    print("Fetching boards …")
    limit = args.board_limit or 250
    boards = paginate("boards", token, page_size=25, limit=limit)
    print(f"Found {len(boards)} board(s).")

    boards_with_images: dict[str, list[Path]] = {}

    for board in boards:
        board_id = board.get("id", "")
        board_name = board.get("name", board_id)
        print(f"\nBoard: {board_name!r}")

        pins = paginate(
            f"boards/{board_id}/pins",
            token,
            page_size=100,
            limit=args.pins_per_board,
        )
        print(f"  {len(pins)} pin(s)")

        slug = safe_slug(board_name)
        board_dir = images_dir / slug
        board_dir.mkdir(exist_ok=True)

        saved: list[Path] = []
        for pin in pins:
            pin_id = pin.get("id", str(len(saved)))
            url = best_image_url(pin)
            if not url:
                continue

            ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
            if ext not in ("jpg", "jpeg", "png", "webp"):
                ext = "jpg"

            img_path = board_dir / f"{pin_id}.{ext}"
            if img_path.exists() or download_image(url, img_path):
                saved.append(img_path)

        print(f"  {len(saved)} image(s) saved → {board_dir}")
        if saved:
            boards_with_images[board_id] = saved

    if not boards_with_images:
        print("No images downloaded. Check your token and board privacy settings.")
        sys.exit(1)

    board_ids = list(boards_with_images)
    rng = random.Random(args.seed)

    # ---- Positive pairs (same board) ------------------------------------
    positive_pairs: list[tuple[Path, Path]] = []
    for bid, paths in boards_with_images.items():
        if len(paths) < 2:
            continue
        pool = list(paths)
        rng.shuffle(pool)
        for i in range(0, min(len(pool) - 1, args.pairs_per_board * 2), 2):
            positive_pairs.append((pool[i], pool[i + 1]))

    rng.shuffle(positive_pairs)

    # ---- Negative pairs (different boards) ------------------------------
    neg_target = max(1, int(len(positive_pairs) * args.neg_ratio))
    negative_pairs: list[tuple[Path, Path]] = []

    if len(board_ids) < 2:
        print(
            "Only one board with images — cannot generate cross-board negatives.\n"
            "All pairs will be positive (same-board). Negatives skipped."
        )
    else:
        attempts = 0
        while len(negative_pairs) < neg_target and attempts < neg_target * 10:
            bid_a, bid_b = rng.sample(board_ids, 2)
            path_a = rng.choice(boards_with_images[bid_a])
            path_b = rng.choice(boards_with_images[bid_b])
            negative_pairs.append((path_a, path_b))
            attempts += 1

    # ---- Write CSV -------------------------------------------------------
    with pairs_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_path", "right_path", "label"])
        for left, right in positive_pairs:
            writer.writerow([left, right, 1])
        for left, right in negative_pairs:
            writer.writerow([left, right, 0])

    print(
        f"\nWrote {len(positive_pairs)} positive + {len(negative_pairs)} negative pairs"
        f"  →  {pairs_csv}"
    )
    print(
        f"\nNext step:\n"
        f"  venv/bin/python scripts/train_polyvore.py \\\n"
        f"    --pairs-csv {pairs_csv} \\\n"
        f"    --epochs 5"
    )


if __name__ == "__main__":
    main()
