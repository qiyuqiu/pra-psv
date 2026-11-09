"""Portable adapter of the reference pre-execution gate for AgentDojo v1.2.2.

Install AgentDojo separately. Pass its actor pipeline and the released AttriGuard
module explicitly; this module neither starts a model server nor downloads code.
"""
import ast
import copy
from ..psv.core import can_review, content, digest


def visible_messages_for_call(messages, processed_calls, results):
    current = dict(messages[-1])
    current["tool_calls"] = list(processed_calls)
    return [*messages[:-1], current, *results]


class AttriGuardReviewer:
    def __init__(self, author_module, actor, attenuation_llm, judge_llm):
        self.author = author_module
        self.actor = actor
        self.loop = author_module.AttriGuardExecutionLoop(
            llm=actor, attenuation_llm=attenuation_llm, judge_llm=judge_llm,
            attenuation_level=2, survival_mode="fuzzy", max_iters=15)
        self.cache = {}

    def shadow(self, query, runtime, env, messages):
        from agentdojo.types import text_content_block_from_string as txt
        history = copy.deepcopy(messages[:-1])
        has = False
        for index, message in enumerate(history):
            if message["role"] != "tool":
                continue
            name = getattr(message.get("tool_call"), "function", None)
            if name and name not in self.author.EXTERNAL_OUTPUT_TOOLS:
                continue
            value = content(message.get("content"))
            if not value.strip():
                continue
            has = True
            key = digest(dict(index=index, name=name, content=value, error=message.get("error")))
            if key not in self.cache:
                self.cache[key] = self.loop._attenuate_observation(value, runtime, env)
            message["content"] = [txt(self.cache[key])]
        self.author._normalize_messages(history)
        if not has:
            return None
        last = []
        for message in reversed(messages[:-1]):
            if message["role"] != "tool":
                break
            last.append(message)
        if last and all(not content(m.get("content")).strip() for m in last):
            return None
        output = self.actor.query(query, runtime, env, history, {})[3]
        return output[-1].get("tool_calls") or []

    def decide(self, call, shadow_calls, query, runtime, env):
        if shadow_calls is None:
            raise ValueError("Missing shadow calls for selected review")
        return bool(self.loop._fuzzy_survive(call, shadow_calls, query, runtime, env))


class AgentDojoGate:
    """Create a new instance for each episode; use instead of ToolExecution.

    Pipeline-compatible query API. The caller owns actor generation, iteration,
    benchmark evaluation, request logging, and resource allocation.
    """
    def __init__(self, pra, *, router=None, state_encoder=None, reviewer=None):
        if router is not None and (state_encoder is None or reviewer is None):
            raise ValueError("PSV requires both a state encoder and an actual reviewer")
        self.pra = pra
        self.router = router
        self.encoder = state_encoder
        self.reviewer = reviewer
        self.reviews_used = 0
        self.events = []

    def query(self, query, runtime, env, messages=(), extra_args=None):
        from agentdojo.types import ChatToolResultMessage, text_content_block_from_string as txt
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str, is_string_list
        from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
        calls = messages[-1].get("tool_calls") or []
        processed, results = [], []
        shadow_ready, shadow_calls = False, None
        for call in calls:
            processed.append(call)
            if call.function not in runtime.functions:
                if call.function != EMPTY_FUNCTION_NAME:
                    error = f"ToolNotFoundError: {call.function}"
                    results.append(ChatToolResultMessage(role="tool", content=[txt(error)],
                        tool_call_id=call.id, tool_call=call, error=error))
                self.events.append(dict(tool=call.function, selected_review=False,
                                        final_allow=False, invalid_or_empty_tool=True))
                continue
            visible = visible_messages_for_call(messages, processed, results)
            score, count, tokens, embedding = self.pra.score(visible)
            allow, selected, prediction = score < self.pra.threshold, False, None
            reasons = []
            if self.router is not None:
                eligible = can_review(messages, self.reviewer.author.EXTERNAL_OUTPUT_TOOLS)
                pack = None
                if eligible and self.reviews_used < 2:
                    pack = self.encoder.pack(query, call, visible, runtime, score,
                                             count, tokens, embedding, self.reviews_used)
                decision, prediction = self.router.select_review(tool=call.function,
                    observation_eligible=eligible, pra_score=score,
                    reviews_used=self.reviews_used, pack=pack)
                selected, reasons = decision.review, list(decision.reasons)
                if selected:
                    self.reviews_used += 1
                    if not shadow_ready:
                        shadow_calls = self.reviewer.shadow(query, runtime, env, messages)
                        shadow_ready = True
                    if shadow_calls is None:
                        raise ValueError("Eligibility/reviewer disagreement")
                    allow = self.reviewer.decide(call, shadow_calls, query, runtime, env)
            if allow:
                args = call.args
                for key, value in args.items():
                    if isinstance(value, str) and is_string_list(value):
                        args[key] = ast.literal_eval(value)
                result, error = runtime.run_function(env, call.function, args)
                message = ChatToolResultMessage(role="tool", content=[txt(tool_result_to_str(result))],
                    tool_call_id=call.id, tool_call=call, error=error)
            elif selected:
                message = self.reviewer.author._blocked_tool_result(call, query)
            else:
                # This literal is retained because it is visible to subsequent actor turns.
                message = ChatToolResultMessage(role="tool", content=[txt("Action blocked by dh_study_pra.")],
                    tool_call_id=call.id, tool_call=call, error="dh_study_pra blocked")
            self.events.append(dict(tool=call.function, pra_score=score,
                selected_review=selected, final_allow=allow,
                executed=allow and not message.get("error"), prediction=prediction,
                reviews_used=self.reviews_used, review_reasons=reasons))
            results.append(message)
        return query, runtime, env, [*messages, *results], extra_args or {}

