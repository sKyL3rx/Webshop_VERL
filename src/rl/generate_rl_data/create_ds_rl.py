import os
import argparse
import requests
import pandas as pd
from tqdm import tqdm

def fetch_initial_prompt(goal_idx: int, base_url: str):
    session_id = None
    try:
        res = requests.post(f"{base_url}/reset", json={"goal_idx": goal_idx}, timeout=10)
        res.raise_for_status()
        data = res.json()

        session_id = data["session_id"]
        instruction = data.get("original_instruction", "").strip()

        requests.post(f"{base_url}/close", json={"session_id": session_id}, timeout=5)

        return instruction, goal_idx
    except Exception as e:
        print(f"\n[Lỗi] Không thể fetch goal_idx {goal_idx}: {e}")
        return None

def build_dataset_split(start_idx: int, end_idx: int, base_url: str) -> pd.DataFrame:
    rows = []
    goal_range = range(start_idx, end_idx + 1)

    print(f"Bắt đầu fetch data từ {start_idx} đến {end_idx}...")
    for gid in tqdm(goal_range):
        result = fetch_initial_prompt(gid, base_url)
        if result is None:
            continue

        instruction, goal_idx = result

        row = {
            "data_source": "webshop",
            "prompt": [{"role": "user", "content": instruction}],
            "ability": "web-navigation",
            "goal_idx": goal_idx,
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": str(gid)}
            },
            "extra_info": {
                "goal_idx": gid
            }
        }
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/workspace/data")
    parser.add_argument("--total_goals", type=int, default=11818)
    parser.add_argument("--url", type=str, default="http://localhost:8001")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n--- TẠO TẬP VALIDATION ---")
    val_df = build_dataset_split(500, 1499, args.url)
    val_path = os.path.join(args.out_dir, "webshop_val.parquet")
    val_df.to_parquet(val_path, index=False)
    print(f"=> Đã lưu Val set: {val_path} ({len(val_df)} rows)")

    print("\n--- TẠO TẬP TRAIN ---")
    train_df = build_dataset_split(1500, args.total_goals, args.url)
    train_path = os.path.join(args.out_dir, "webshop_train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"=> Đã lưu Train set: {train_path} ({len(train_df)} rows)")

if __name__ == "__main__":
    main()
