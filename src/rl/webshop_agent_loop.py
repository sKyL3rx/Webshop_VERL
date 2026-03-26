import asyncio
import logging
import re
import sys
import time
from typing import Any, Dict, List
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = True

ACTION_RE = re.compile(
    r"(search\[[^\]]*\]|click\[[^\]]+\]|finish)",
    re.I | re.S,
)


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def clean_observation_for_prompt(obs_text: str) -> str:
    """
    Hỗ trợ cả:
    - raw WebShop obs kiểu nhiều dòng cũ
    - obs đã được env_server clean thành chuỗi pipe-separated
    """
    obs_text = (obs_text or "").strip()
    if not obs_text:
        return "(Empty observation)"

    if "|" in obs_text and "[button]" not in obs_text and "Instruction:" not in obs_text:
        return obs_text

    lines = obs_text.split("\n")
    clean_lines = []
    skip_next = False

    for line in lines:
        stripped = line.strip()

        if stripped == "WebShop":
            continue

        if stripped == "Instruction:":
            skip_next = True
            continue

        if skip_next:
            skip_next = False
            continue

        clean_lines.append(line)

    out = "\n".join(clean_lines).strip()
    return out or "(Empty observation)"


def build_obs_preview(next_obs: str, actual_instruction: str) -> str:
    next_obs = (next_obs or "").strip()
    if not next_obs:
        return "Page updated."

    # new env server đã trả clean obs dạng 1 dòng
    if "|" in next_obs and "[button]" not in next_obs and "Instruction:" not in next_obs:
        preview_text = next_obs
        return preview_text[:200] + ("..." if len(preview_text) > 200 else "")

    lines = next_obs.split("\n")
    ignore_prefixes = [
        "Instruction:",
        actual_instruction,
        "[button] Back to Search",
        "[button] < Prev",
        "[button] Next >",
        "Page ",
    ]

    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("You have clicked"):
            clean_lines.append(stripped)
            continue
        if stripped == "WebShop":
            continue
        if any(stripped.startswith(prefix) for prefix in ignore_prefixes):
            continue
        clean_lines.append(stripped)

    if clean_lines:
        preview_text = " | ".join(clean_lines[:2])
        return preview_text[:200] + ("..." if len(preview_text) > 200 else "")

    return "Page updated."


def extract_title_casing_map(obs_text: str) -> Dict[str, str]:
    title_casing_map: Dict[str, str] = {}

    for block in obs_text.split("[button] "):
        lines = block.split("\n")
        if len(lines) >= 2 and "[button_]" in lines[0]:
            original_title = lines[1].strip()
            if original_title:
                title_casing_map[original_title.lower()] = original_title

    return title_casing_map


def format_valid_actions(valids: List[str], obs_text: str) -> str:
    """
    Hỗ trợ cả 2 kiểu valid actions:
    - cũ: click[item - ...]
    - mới từ env_server: click[ASIN], click[description], click[< prev], ...
    """
    title_casing_map = extract_title_casing_map(obs_text or "")

    out = []
    added_search = False

    for act in dedupe_keep_order(valids or []):
        act = (act or "").strip()
        if not act:
            continue

        if act.startswith("search["):
            if not added_search:
                out.append("search[<your transformed query>]")
                added_search = True
            continue

        if act.startswith("click[item - "):
            raw_title = act[len("click[item - "):-1]
            exact_title = title_casing_map.get(raw_title.lower(), raw_title)
            out.append(f"click[item - {exact_title}]")
            continue

        out.append(act)

    if not out:
        return "- finish"

    return "\n".join(f"- {x}" for x in dedupe_keep_order(out))


def build_user_turn(
    prefix: str,
    instruction: str,
    observation: str,
    valid_actions_str: str,
    history_text: str,
) -> str:
    return """{prefix}

Instruction:
{instruction}

Current observation:
{observation}

Valid actions:
{valid_actions}

History:
{history_text}

Return exactly ONE next action.
- If a search action is available, you may output a free-form action of the form search[your query].
- Otherwise, output exactly one click[...] action from the valid actions list.""".format(
        prefix=prefix,
        instruction=instruction,
        observation=observation,
        valid_actions=valid_actions_str,
        history_text=history_text,
    )


def extract_executable_action(text: str) -> str:
    if not text:
        return ""
    m = ACTION_RE.search(text.strip())
    if not m:
        return ""
    return m.group(1).strip()


class RemoteWebShopClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _normalize_reset_payload(self, data: dict) -> dict:
        """
        Env server mới trả:
          {ok, session_id, obs, valid, reward, done, original_instruction}
        Agent loop cũ kỳ vọng:
          {session_id, obs, info={"valid": ...}, instruction, ...}
        """
        obs = data.get("obs", "") or ""
        valid = data.get("valid", []) or []
        instruction = (
            data.get("instruction")
            or data.get("original_instruction")
            or ""
        )

        return {
            **data,
            "obs": obs,
            "info": {"valid": valid},
            "instruction": instruction,
        }

    def _normalize_step_payload(self, data: dict, raw_action: str) -> dict:
        """
        Env server mới trả:
          {ok, session_id, obs, valid, reward, done, original_instruction}
        Agent loop cũ kỳ vọng:
          {obs, info={"valid": ...}, executed_action, ...}
        """
        obs = data.get("obs", "") or ""
        valid = data.get("valid", []) or []
        instruction = (
            data.get("instruction")
            or data.get("original_instruction")
            or ""
        )

        # reviews có thể trả obs="" nhưng valid vẫn còn < prev/back to search>
        if not obs and valid:
            obs = "(Empty page content. Use one of the valid actions to continue.)"

        return {
            **data,
            "obs": obs,
            "info": {"valid": valid},
            "instruction": instruction,
            "executed_action": data.get("executed_action", raw_action),
        }

    async def reset(self, client: httpx.AsyncClient, goal_idx: int) -> dict:
        last_err = None
        for attempt in range(5):
            try:
                r = await client.post("/reset", json={"goal_idx": int(goal_idx)})
                if r.status_code == 503:
                    data = r.json()
                    logger.warning(
                        "[env/reset_busy] goal_idx=%s attempt=%s payload=%r",
                        goal_idx,
                        attempt + 1,
                        data,
                    )
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue

                r.raise_for_status()
                data = r.json()
                if not data.get("ok", False):
                    raise RuntimeError(data.get("error", "reset_failed"))
                return self._normalize_reset_payload(data)
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                last_err = e
                logger.warning(
                    "[env/reset_retry] goal_idx=%s attempt=%s err=%r",
                    goal_idx,
                    attempt + 1,
                    e,
                )
                await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))

        if last_err is not None:
            raise last_err
        raise RuntimeError("reset_failed_after_retries")

    async def step(self, client: httpx.AsyncClient, session_id: str, raw_action: str) -> dict:
        last_err = None
        for attempt in range(3):
            try:
                r = await client.post(
                    "/step",
                    json={"session_id": session_id, "raw_action": raw_action},
                )

                # env server của bạn trả 400 cho invalid_action / unknown_session
                if r.status_code in (400, 404):
                    data = r.json()
                    return self._normalize_step_payload(
                        {
                            "ok": False,
                            "session_id": session_id,
                            "obs": data.get("obs", ""),
                            "valid": data.get("valid", []),
                            "reward": float(data.get("reward", 0.0)),
                            "done": bool(data.get("done", True)),
                            "error": data.get("error", f"http_{r.status_code}"),
                            "original_instruction": data.get("original_instruction", ""),
                        },
                        raw_action=raw_action,
                    )

                r.raise_for_status()
                data = r.json()
                return self._normalize_step_payload(data, raw_action=raw_action)

            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                last_err = e
                logger.warning(
                    "[env/step_retry] session=%s action=%r attempt=%s err=%r",
                    session_id,
                    raw_action,
                    attempt + 1,
                    e,
                )
                await asyncio.sleep(min(1.0 * (attempt + 1), 4.0))

        if last_err is not None:
            raise last_err
        raise RuntimeError("step_failed_after_retries")

    async def close(self, client: httpx.AsyncClient, session_id: str) -> None:
        try:
            await client.post("/close", json={"session_id": session_id})
        except Exception:
            pass


@register("webshop_agent")
class WebShopAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        dataset_config=None,
        data_config=None,
        **kwargs,
    ):
        if dataset_config is None:
            dataset_config = data_config

        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            dataset_config=dataset_config,
            **kwargs,
        )

        cfg = getattr(trainer_config, "config", trainer_config)

        rollout_cfg = None
        try:
            rollout_cfg = cfg.actor_rollout_ref.rollout
        except Exception:
            try:
                rollout_cfg = cfg.rollout
            except Exception:
                rollout_cfg = None

        mt_cfg = getattr(rollout_cfg, "multi_turn", None)
        mt_max_turns = None
        if mt_cfg not in (None, True, False):
            mt_max_turns = getattr(mt_cfg, "max_assistant_turns", None)

        self.webshop_base_url = kwargs.get("webshop_base_url", "http://127.0.0.1:8001")
        self.webshop_timeout = httpx.Timeout(
            connect=float(kwargs.get("webshop_connect_timeout", 5.0)),
            read=float(kwargs.get("webshop_read_timeout", 300.0)),
            write=float(kwargs.get("webshop_write_timeout", 30.0)),
            pool=float(kwargs.get("webshop_pool_timeout", 60.0)),
        )
        self.webshop_limits = httpx.Limits(
            max_keepalive_connections=int(kwargs.get("webshop_max_keepalive_connections", 20)),
            max_connections=int(kwargs.get("webshop_max_connections", 100)),
        )

        self.remote_env = RemoteWebShopClient(base_url=self.webshop_base_url)

        self.max_turns = int(kwargs.get("max_turns", mt_max_turns or 15))
        self.response_length = int(
            getattr(
                rollout_cfg,
                "response_length",
                getattr(getattr(cfg, "data", None), "max_response_length", 512),
            )
        )
        self.max_history = int(kwargs.get("max_history", 6))
        self.prompt_prefix = kwargs.get("prompt_prefix", "You are a shopping agent.")

        self.debug_rollout_every = int(kwargs.get("debug_rollout_every", 1))
        self.debug_max_chars = int(kwargs.get("debug_max_chars", 400))
        self.debug_print_valid_actions = bool(kwargs.get("debug_print_valid_actions", False))

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        goal_idx = kwargs.get("goal_idx")
        if goal_idx is None:
            raise ValueError("Dataset row must contain goal_idx for WebShop agent loop.")

        debug_this_episode = (
            self.debug_rollout_every > 0 and int(goal_idx) % self.debug_rollout_every == 0
        )
        trajectory_log: List[Dict[str, Any]] = []

        print(f"[rollout/alive] goal_idx={goal_idx}", flush=True)
        logger.warning("[rollout/alive] goal_idx=%s", goal_idx)

        async with httpx.AsyncClient(
            base_url=self.webshop_base_url,
            timeout=self.webshop_timeout,
            limits=self.webshop_limits,
            trust_env=False,
        ) as client:
            t_reset0 = time.perf_counter()
            reset_data = await self.remote_env.reset(client, int(goal_idx))
            reset_latency = time.perf_counter() - t_reset0

            session_id = reset_data["session_id"]
            obs = reset_data["obs"]
            info = reset_data["info"]
            actual_instruction = reset_data.get("instruction", "")

            try:
                history_lines: List[str] = []
                history_text = "(No history - Start of session)"

                clean_obs = clean_observation_for_prompt(obs)
                valid_actions_str = format_valid_actions(info.get("valid", []), obs)

                if debug_this_episode:
                    logger.warning(
                        "[rollout/reset] goal_idx=%s session=%s latency=%.3fs instruction=%r obs=%r",
                        goal_idx,
                        session_id,
                        reset_latency,
                        actual_instruction[: self.debug_max_chars],
                        clean_obs[: self.debug_max_chars],
                    )
                    if self.debug_print_valid_actions:
                        logger.warning(
                            "[rollout/valid_actions] goal_idx=%s valid=%r",
                            goal_idx,
                            info.get("valid", []),
                        )

                initial_user_content = build_user_turn(
                    prefix=self.prompt_prefix,
                    instruction=actual_instruction,
                    observation=clean_obs,
                    valid_actions_str=valid_actions_str,
                    history_text=history_text,
                )

                messages = [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": initial_user_content},
                ]
                prompt_ids = await self.apply_chat_template(messages)
                running_prompt_ids = list(prompt_ids)

                response_ids: List[int] = []
                response_mask: List[int] = []
                response_logprobs: List[float] = []
                track_logprobs = False

                request_id = uuid4().hex
                reward_score = 0.0
                env_calls = 0
                assistant_turns = 0
                user_turns = 0
                start_time = time.time()

                for step_idx in range(self.max_turns):
                    if len(response_mask) >= self.response_length:
                        logger.warning(
                            "[rollout/stop_response_limit] goal_idx=%s step=%s len=%s limit=%s",
                            goal_idx,
                            step_idx,
                            len(response_mask),
                            self.response_length,
                        )
                        break

                    model_out = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=running_prompt_ids,
                        sampling_params=sampling_params,
                    )

                    turn_token_ids = list(model_out.token_ids)
                    if not turn_token_ids:
                        logger.warning(
                            "[rollout/no_tokens] goal_idx=%s step=%s",
                            goal_idx,
                            step_idx,
                        )
                        break

                    running_prompt_ids += turn_token_ids
                    response_ids += turn_token_ids
                    response_mask += [1] * len(turn_token_ids)
                    assistant_turns += 1

                    if getattr(model_out, "log_probs", None) is not None:
                        track_logprobs = True
                        response_logprobs += list(model_out.log_probs)
                    elif track_logprobs:
                        response_logprobs += [0.0] * len(turn_token_ids)

                    raw_response_text = self.tokenizer.decode(
                        turn_token_ids,
                        skip_special_tokens=True,
                    ).strip()

                    executed_action = extract_executable_action(raw_response_text)
                    assistant_content = executed_action if executed_action else "finish"
                    messages.append({"role": "assistant", "content": assistant_content})

                    if debug_this_episode:
                        logger.warning(
                            "[rollout/model] goal_idx=%s step=%s text=%r action=%r",
                            goal_idx,
                            step_idx,
                            raw_response_text[: self.debug_max_chars],
                            executed_action,
                        )

                    if not executed_action or executed_action == "finish":
                        trajectory_log.append(
                            {
                                "step": step_idx,
                                "action": executed_action,
                                "reward": reward_score,
                                "done": True,
                                "obs_preview": "finish_or_no_action",
                            }
                        )
                        break

                    t_step0 = time.perf_counter()
                    step_data = await self.remote_env.step(
                        client=client,
                        session_id=session_id,
                        raw_action=executed_action,
                    )
                    step_latency = time.perf_counter() - t_step0
                    env_calls += 1

                    if not step_data.get("ok", False):
                        reward_score = float(step_data.get("reward", 0.0))
                        trajectory_log.append(
                            {
                                "step": step_idx,
                                "action": executed_action,
                                "reward": reward_score,
                                "done": True,
                                "obs_preview": f"env_error:{step_data.get('error', 'unknown')}",
                            }
                        )
                        logger.warning(
                            "[rollout/env_error] goal_idx=%s step=%s latency=%.3fs payload=%r",
                            goal_idx,
                            step_idx,
                            step_latency,
                            step_data,
                        )
                        break

                    action_for_history = step_data.get("executed_action", executed_action)
                    next_obs = step_data["obs"]
                    next_info = step_data["info"]
                    reward_score = float(step_data.get("reward", 0.0))
                    done = bool(step_data.get("done", False))

                    obs_preview = build_obs_preview(next_obs, actual_instruction)
                    history_lines.append(
                        f"Step {step_idx}: {action_for_history} -> Result: {obs_preview}"
                    )
                    history_text = "\n".join(history_lines[-self.max_history:])

                    trajectory_log.append(
                        {
                            "step": step_idx,
                            "action": action_for_history,
                            "reward": reward_score,
                            "done": done,
                            "obs_preview": obs_preview,
                            "step_latency_sec": round(step_latency, 3),
                        }
                    )

                    if debug_this_episode:
                        logger.warning(
                            "[rollout/env] goal_idx=%s step=%s latency=%.3fs reward=%.4f done=%s action=%r obs=%r",
                            goal_idx,
                            step_idx,
                            step_latency,
                            reward_score,
                            done,
                            action_for_history,
                            obs_preview[: self.debug_max_chars],
                        )

                    if done:
                        break

                    clean_next_obs = clean_observation_for_prompt(next_obs)
                    next_valid_actions_str = format_valid_actions(
                        next_info.get("valid", []),
                        next_obs,
                    )

                    user_content = build_user_turn(
                        prefix=self.prompt_prefix,
                        instruction=actual_instruction,
                        observation=clean_next_obs,
                        valid_actions_str=next_valid_actions_str,
                        history_text=history_text,
                    )

                    logger.warning(
                        "[rollout/user_content_stats] goal_idx=%s step=%s chars=%s lines=%s obs_chars=%s valid_chars=%s history_chars=%s",
                        goal_idx,
                        step_idx,
                        len(user_content),
                        len(user_content.splitlines()),
                        len(clean_next_obs),
                        len(next_valid_actions_str),
                        len(history_text),
                    )

                    add_messages = [{"role": "user", "content": user_content}]
                    user_response_ids = await self.apply_chat_template(
                        add_messages,
                        remove_system_prompt=True,
                    )

                    logger.warning(
                        "[rollout/user_token_budget] goal_idx=%s step=%s user_tokens=%s",
                        goal_idx,
                        step_idx,
                        len(user_response_ids),
                    )

                    if len(response_mask) + len(user_response_ids) > self.response_length:
                        logger.warning(
                            "[rollout/response_limit] goal_idx=%s step=%s current_mask=%s user_tokens=%s limit=%s",
                            goal_idx,
                            step_idx,
                            len(response_mask),
                            len(user_response_ids),
                            self.response_length,
                        )
                        break

                    messages.extend(add_messages)
                    running_prompt_ids += user_response_ids
                    response_ids += user_response_ids
                    response_mask += [0] * len(user_response_ids)

                    if track_logprobs:
                        response_logprobs += [0.0] * len(user_response_ids)

                    user_turns += 1
                    obs, info = next_obs, next_info

                metrics = AgentLoopMetrics(
                    generate_sequences=time.time() - start_time,
                    tool_calls=float(env_calls),
                )

                if debug_this_episode:
                    logger.warning(
                        "[rollout/final] goal_idx=%s final_reward=%.4f turns=%s trajectory=%s",
                        goal_idx,
                        reward_score,
                        user_turns + assistant_turns + 1,
                        trajectory_log,
                    )

                return AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=response_mask[: self.response_length],
                    response_logprobs=(
                        response_logprobs[: self.response_length]
                        if track_logprobs
                        else None
                    ),
                    reward_score=reward_score,
                    num_turns=user_turns + assistant_turns + 1,
                    metrics=metrics,
                    extra_fields={},
                )
            finally:
                await self.remote_env.close(client, session_id)
