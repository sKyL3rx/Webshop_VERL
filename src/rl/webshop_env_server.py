import argparse
import re
import threading
from queue import LifoQueue, Empty
from typing import Dict, List, Optional
from uuid import uuid4

from flask import Flask, jsonify, request
from webshop.baseline_models.env import WebEnv

CLICK_ACTION_RE = re.compile(r"^click\[(.+)\]$", re.I)
SEARCH_ACTION_RE = re.compile(r"^search\[(.*)\]$", re.I)


class SFTArgs:
    state_format = "text_rich"
    click_item_name = 1
    get_image = 0
    num_prev_obs = 0
    num_prev_actions = 0
    human_goals = 1
    num = None
    step_limit = 100
    extra_search_path = ""
    harsh_reward = 0
    go_to_item = 0
    go_to_search = 0
    ban_buy = 0

def is_search_action(action: str) -> bool:
    return bool(action) and bool(SEARCH_ACTION_RE.fullmatch(action.strip()))


def is_click_action(action: str) -> bool:
    return bool(action) and bool(CLICK_ACTION_RE.fullmatch(action.strip()))


def asin_to_title_click_if_possible(env: WebEnv, action: str, valid_actions: List[str]) -> str:
    action = normalize_click_action(action)
    m = CLICK_ACTION_RE.fullmatch(action)
    if not m:
        return action

    arg = m.group(1).strip()
    if not re.fullmatch(r"[A-Z0-9]{10}", arg):
        return action

    asin2name = getattr(env, "asin2name", {})
    title = asin2name.get(arg.lower())
    if not title:
        return action

    candidate = f"click[item - {title}]"
    if candidate in valid_actions:
        return candidate

    for v in valid_actions:
        if v.lower() == candidate.lower():
            return v

    return action


def normalize_click_action(action: str) -> str:
    if not action:
        return action
    action = action.strip()
    m = CLICK_ACTION_RE.fullmatch(action)
    if not m:
        return action

    arg = m.group(1).strip()

    if re.fullmatch(r"[a-zA-Z0-9]{10}", arg):
        return f"click[{arg.upper()}]"

    return action

def adapt_action_to_env_format(env: WebEnv, raw_action: str, valid_actions: List[str]) -> str:
    raw_action = (raw_action or "").strip()

    if is_search_action(raw_action):
        return raw_action

    if not is_click_action(raw_action):
        return raw_action

    norm_action = normalize_click_action(raw_action)

    if norm_action in valid_actions:
        return norm_action
    
    if getattr(env, "click_item_name", 0):
        adapted = asin_to_title_click_if_possible(env, norm_action, valid_actions)
        if adapted in valid_actions:
            return adapted
        
    norm_lower = norm_action.lower()
    for v in valid_actions:
        if v.lower() == norm_lower:
            return v

    return norm_action


def can_execute_action(action_for_env: str, info: dict) -> bool:
    valids = info.get("valid", [])

    if is_search_action(action_for_env):
        return any(v.startswith("search[") for v in valids)

    return action_for_env in valids

def build_env(env_split: str) -> WebEnv:
    env = WebEnv(SFTArgs(), split=env_split)
    env.reduce_click = False
    return env


class EnvManager:
    def __init__(self, env_split: str, pool_size: int):
        self.env_split = env_split
        self.pool_size = pool_size
        self.free_envs = LifoQueue(maxsize=pool_size)
        self.sessions: Dict[str, dict] = {}
        self.lock = threading.Lock()

        for _ in range(pool_size):
            self.free_envs.put(build_env(env_split))

    def _acquire_env(self) -> WebEnv:
        try:
            return self.free_envs.get_nowait()
        except Empty:
            return build_env(self.env_split)

    def _release_env(self, env: WebEnv) -> None:
        try:
            self.free_envs.put_nowait(env)
        except Exception:
            try:
                env.close()
            except Exception:
                pass
    
    def reset(self, goal_idx: int) -> dict:
        env = self._acquire_env()
        obs, info = env.reset(int(goal_idx))

        instruction = getattr(env.env, "instruction_text", "") or ""
        if instruction.startswith("Instruction:"):
            instruction = instruction.replace("Instruction:", "", 1).strip()

        session_id = uuid4().hex
        state = {
            "env": env,
            "goal_idx": int(goal_idx),
            "obs": obs,
            "info": info,
            "instruction": instruction,
            "done": False,
        }

        with self.lock:
            self.sessions[session_id] = state

        return {
            "ok": True,
            "session_id": session_id,
            "obs": obs,
            "info": info,
            "instruction": instruction,
            "reward": 0.0,
            "done": False,
        }

    def step(self, session_id: str, raw_action: str) -> dict:
        with self.lock:
            state = self.sessions.get(session_id)
        
        if state is None:
            return {
                "ok": False,
                "error": "unknown_session",
                "session_id": session_id,
                "obs": "",
                "info": {},
                "reward": 0.0,
                "done": True,
                "executed_action": raw_action,
            }

        env = state["env"]
        info = state["info"]

        action_for_env = adapt_action_to_env_format(
            env=env,
            raw_action=raw_action,
            valid_actions=info.get("valid", []),
        )

        if not can_execute_action(action_for_env, info):
            return {
                "ok": False,
                "error": "invalid_action",
                "session_id": session_id,
                "obs": state.get("obs", ""),
                "info": info,
                "reward": 0.0,
                "done": True,
                "executed_action": action_for_env,
            }


        try:
            next_obs, reward, done, next_info = env.step(action_for_env)
        except Exception as e:
            return {
                "ok": False,
                "error": f"env_step_failed: {type(e).__name__}: {e}",
                "session_id": session_id,
                "obs": state.get("obs", ""),
                "info": info,
                "reward": 0.0,
                "done": True,
                "executed_action": action_for_env,
            }

        state["obs"] = next_obs
        state["info"] = next_info
        state["done"] = bool(done)

        return {
            "ok": True,
            "session_id": session_id,
            "obs": next_obs,
            "info": next_info,
            "reward": float(reward),
            "done": bool(done),
            "executed_action": action_for_env,
            "instruction": state["instruction"],
        }
    
    def close(self, session_id: str) -> dict:
        with self.lock:
            state = self.sessions.pop(session_id, None)

        if state is None:
            return {"ok": True, "closed": False}

        env = state["env"]
        self._release_env(env)
        return {"ok": True, "closed": True}

    def health(self) -> dict:
        with self.lock:
            active_sessions = len(self.sessions)

        return {
            "ok": True,
            "env_split": self.env_split,
            "pool_size": self.pool_size,
            "active_sessions": active_sessions,
            "free_envs": self.free_envs.qsize(),
        }

def create_app(env_split: str, pool_size: int) -> Flask:
    app = Flask(__name__)
    manager = EnvManager(env_split=env_split, pool_size=pool_size)

    @app.get("/health")
    def health():
        return jsonify(manager.health())

    @app.post("/reset")
    def reset():
        payload = request.get_json(force=True, silent=True) or {}
        goal_idx = payload.get("goal_idx")
        if goal_idx is None:
            return jsonify({"ok": False, "error": "missing_goal_idx"}), 400
        return jsonify(manager.reset(int(goal_idx)))

    @app.post("/step")
    def step():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id")
        raw_action = payload.get("raw_action", "")

        if not session_id:
            return jsonify({"ok": False, "error": "missing_session_id"}), 400

        result = manager.step(session_id=session_id, raw_action=raw_action)
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    @app.post("/close")
    def close():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "missing_session_id"}), 400
        return jsonify(manager.close(session_id))

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--split", default="train")
    parser.add_argument("--pool-size", type=int, default=2)
    args = parser.parse_args()

    app = create_app(env_split=args.split, pool_size=args.pool_size)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()