import asyncio
import re
import time
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from webshop.baseline_models.env import WebEnv


ACTION_RE = re.compile(
    r"(search\[[^\]]*\]|click\[[^\]]+\]|finish)",
    re.I | re.S,
)
CLICK_ACTION_RE = re.compile(r"^click\[(.+)\]$", re.I)
SEARCH_ACTION_RE = re.compile(r"^search\[(.*)\]$", re.I)

# Process-local pools: key = (env_split, pool_size)
_ENV_POOLS: Dict[Tuple[str, int], asyncio.Queue] = {}
_ENV_POOL_INIT_LOCK = asyncio.Lock()


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


def _build_env(env_split: str) -> WebEnv:
    env = WebEnv(SFTArgs(), split=env_split)
    env.reduce_click = False
    return env


async def get_or_create_env_pool(pool_size: int, env_split: str) -> asyncio.Queue:
    key = (env_split, pool_size)

    if key in _ENV_POOLS:
        return _ENV_POOLS[key]

    async with _ENV_POOL_INIT_LOCK:
        if key in _ENV_POOLS:
            return _ENV_POOLS[key]

        q: asyncio.Queue = asyncio.Queue(maxsize=pool_size)

        # Initialize sequentially to avoid CPU/RAM stampede.
        for _ in range(pool_size):
            env = await asyncio.to_thread(_build_env, env_split)
            await q.put(env)

        _ENV_POOLS[key] = q
        return q


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def is_search_action(action: str) -> bool:
    return bool(action) and bool(SEARCH_ACTION_RE.fullmatch(action.strip()))


def is_click_action(action: str) -> bool:
    return bool(action) and bool(CLICK_ACTION_RE.fullmatch(action.strip()))


def normalize_click_action(action: str) -> str:
    if not action:
        return action
    action = action.strip()
    m = CLICK_ACTION_RE.fullmatch(action)
    if not m:
        return action

    arg = m.group(1).strip()

    # Nếu là ASIN 10 ký tự thì upper lại cho ổn định.
    if re.fullmatch(r"[a-zA-Z0-9]{10}", arg):
        return f"click[{arg.upper()}]"

    return action


def clean_observation_for_prompt(obs_text: str) -> str:
    lines = obs_text.strip().split("\n")
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

    return "\n".join(clean_lines).strip()


def build_obs_preview(next_obs: str, actual_instruction: str) -> str:
    lines = next_obs.strip().split("\n")
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
        return preview_text[:150] + ("..." if len(preview_text) > 150 else "")

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
    title_casing_map = extract_title_casing_map(obs_text)

    out = []
    added_search = False

    for act in dedupe_keep_order(valids):
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


def adapt_action_to_env_format(env: WebEnv, raw_action: str, valid_actions: List[str]) -> str:
    raw_action = raw_action.strip()

    if is_search_action(raw_action):
        return raw_action

    if not is_click_action(raw_action):
        return raw_action

    norm_action = normalize_click_action(raw_action)

    if norm_action in valid_actions:
        return norm_action

    # Nếu env đang dùng click theo item title, thử map ASIN -> title
    if getattr(env, "click_item_name", 0):
        adapted = asin_to_title_click_if_possible(env, norm_action, valid_actions)
        if adapted in valid_actions:
            return adapted

    # Match không phân biệt hoa thường
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


@register("webshop_agent")
class WebShopAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config,
            server_manager,
            tokenizer,
            processor,
            dataset_cls,
            data_config,
            **kwargs,
        )

        mt_cfg = getattr(self.rollout_config, "multi_turn", None)
        mt_max_turns = None
        if mt_cfg not in (None, True, False):
            mt_max_turns = getattr(mt_cfg, "max_assistant_turns", None)

        self.max_turns = int(kwargs.get("max_turns", mt_max_turns or 15))
        self.response_length = int(getattr(self.rollout_config, "response_length", 512))
        self.env_pool_size = int(kwargs.get("env_pool_size", 2))
        self.env_split = kwargs.get("env_split", "train")
        self.max_history = int(kwargs.get("max_history", 6))
        self.prompt_prefix = kwargs.get("prompt_prefix", "You are a shopping agent.")

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        goal_idx = kwargs.get("goal_idx")
        if goal_idx is None:
            raise ValueError("Dataset row must contain goal_idx for WebShop agent loop.")

        env_pool = await get_or_create_env_pool(
            pool_size=self.env_pool_size,
            env_split=self.env_split,
        )

        env = await env_pool.get()
        try:
            obs, info = await asyncio.to_thread(env.reset, int(goal_idx))

            actual_instruction = env.env.instruction_text
            if actual_instruction.startswith("Instruction:"):
                actual_instruction = actual_instruction.replace("Instruction:", "", 1).strip()

            history_lines: List[str] = []
            history_text = "(No history - Start of session)"

            clean_obs = clean_observation_for_prompt(obs)
            valid_actions_str = format_valid_actions(info.get("valid", []), obs)

            initial_user_content = build_user_turn(
                prefix=self.prompt_prefix,
                instruction=actual_instruction,
                observation=clean_obs,
                valid_actions_str=valid_actions_str,
                history_text=history_text,
            )

            messages = [{"role": "user", "content": initial_user_content}]

            # Template prompt đầu tiên đúng 1 lần.
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
                    break

                model_out = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=running_prompt_ids,
                    sampling_params=sampling_params,
                )

                turn_token_ids = list(model_out.token_ids)
                if not turn_token_ids:
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

                messages.append({"role": "assistant", "content": raw_response_text})

                if len(response_mask) >= self.response_length:
                    break

                executed_action = extract_executable_action(raw_response_text)

                # Model không trả được action hợp lệ để parse.
                if not executed_action:
                    reward_score = 0.0
                    break

                # Nếu có synthetic finish thì dừng luôn, không env.step().
                if executed_action == "finish":
                    break

                current_valid_actions = info.get("valid", [])
                action_for_env = adapt_action_to_env_format(
                    env=env,
                    raw_action=executed_action,
                    valid_actions=current_valid_actions,
                )

                if not can_execute_action(action_for_env, info):
                    reward_score = 0.0
                    break

                try:
                    next_obs, reward, done, next_info = await asyncio.to_thread(
                        env.step,
                        action_for_env,
                    )
                except Exception:
                    reward_score = 0.0
                    break

                env_calls += 1
                reward_score = float(reward)

                obs_preview = build_obs_preview(next_obs, actual_instruction)
                history_lines.append(
                    f"Step {step_idx}: {action_for_env} -> Result: {obs_preview}"
                )
                history_text = "\n".join(history_lines[-self.max_history:])

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

                add_messages = [{"role": "user", "content": user_content}]

                # Chỉ template delta user/env turn mới.
                user_response_ids = await self.apply_chat_template(
                    add_messages,
                    remove_system_prompt=True,
                )

                if len(response_mask) + len(user_response_ids) > self.response_length:
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
            await env_pool.put(env)