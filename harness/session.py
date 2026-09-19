"""run_session (Algorithm 2 line 5): one tutoring session between a teacher
instance and a student instance under Algorithm 1.

Fixed text (curriculum phrases, item stems, feedback templates) is generated
by the harness; the teacher's explanatory turns are generated live
(Section V-C). The student never sees the specification or the answer key:
the key exists only as the `key` variable below and in the `meta` side
channel to the client, which is never part of a prompt."""
from __future__ import annotations
from typing import Dict, List, Optional
from .content.base import ContentSet, Item
from .learner import History, Interaction, LearnerModel
from .parsing import parse_answer
from .policy import Policy
from .router import Router, Instance
from .telemetry import Telemetry

STUDENT_SYSTEM = ("You are a learner in a tutoring session. Read what your tutor explains, then answer each "
                  "question. Keep any reasoning to a few sentences. Always finish with one final line of the form "
                  "'ANSWER: <letter>' and write nothing after it. If your reasoning runs long, stop and give the answer line.")
TEACHER_SYSTEM = ("You are a tutor. Teach the topic you are asked to teach from your own understanding, in your "
                  "own words, for a learner who has never seen it. Do not state answers to specific test questions. ")
FEEDBACK = {True: "Your previous answer was correct.", False: "Your previous answer was not correct."}
TEACH_PROMPT = "TEACH: Explain the component '{name}' to your student. Cover what it is, the rule, and one worked illustration."


def teach(router: Router, teacher: Instance, name: str, cap: int) -> str:
    comp = router.send(teacher, TEACH_PROMPT.format(name=name), cap)
    return comp.text


def present(router: Router, student: Instance, item: Item, preface: str, cap: int, telemetry: Telemetry,
            base: Dict, phase: str, h: History, learner: Optional[LearnerModel], action: Optional[str] = None,
            p_next: float = None) -> Interaction:
    prompt, key, tagmap = item.render(student.rng)
    msg = (preface + "\n\n" if preface else "") + prompt
    comp = router.send(student, msg, cap, meta={"key": key, "item_id": item.item_id})
    label, status = parse_answer(comp.text)
    correct = status != "failure" and label == key
    misc = tagmap.get(label) if (label and not correct) else None
    inter = Interaction(item.component, item.family, correct, status, comp.latency_ms,
                        student.context_length(), len(h.interactions) + 1, item.difficulty)
    if phase == "instruction":
        h.interactions.append(inter)
        if learner is not None:
            learner.observe(h, inter)
    logbase = {k: v for k, v in base.items() if not k.startswith("_")}
    telemetry.log(**logbase, student_model=student.served_model or student.model, component=item.component,
                  item_id=item.item_id, family=item.family, held_out=item.held_out,
                  correct=correct, distractor=label if not correct else None, misconception=misc,
                  parse_status=status, latency_ms=comp.latency_ms, tokens_in=comp.tokens_in,
                  tokens_out=comp.tokens_out, context_position=len(student.messages),
                  context_length=student.context_length(), phase=phase, action=action, p_next=p_next,
                  attempt_index=h.attempts(item.component), answer_label=label)
    return inter


def run_session(router: Router, teacher: Instance, student: Instance, content: ContentSet, tau: float,
                learner: LearnerModel, cfg, telemetry: Telemetry, base: Dict, n_assess: int = None) -> Dict:
    """Returns a summary dict: outcome, interactions, mastered components,
    items-to-mastery per component (None if censored), post and transfer accuracy."""
    n_assess = n_assess or getattr(cfg, "n_assess", 4)
    student.system = STUDENT_SYSTEM
    if not teacher.system:
        teacher.system = TEACHER_SYSTEM
    base = dict(base)
    h = History()
    policy = Policy(content, tau, learner, cfg.attempt_cap, cfg.review_interval, base.get("_condition", {}))
    c = content.topo_order()[0]
    preface = teach(router, teacher, content.graph()[c].name, cfg.teach_token_cap)
    outcome = "cap"
    items_to_mastery: Dict[str, Optional[int]] = {}
    first_mastered: Optional[str] = None
    feedback = ""
    next_item_component, phase = c, "instruction"
    action_kind = "START"
    p_prev = None
    while len(h.interactions) < cfg.session_cap:
        item = content.sample_item(next_item_component, student.rng)
        pre = (feedback + "\n\n" + preface).strip() if (feedback or preface) else ""
        present(router, student, item, pre, cfg.answer_token_cap, telemetry, base, phase, h, learner, action_kind, p_prev)
        feedback, preface = FEEDBACK[h.interactions[-1].correct], ""
        act = policy.decide(h, c)
        p_prev, action_kind = act.p, act.kind
        if c in policy.mastered and c not in items_to_mastery:
            items_to_mastery[c] = h.attempts(c)
            first_mastered = first_mastered or c
        if act.kind == "TERMINATE":
            outcome = act.reason
            break
        if act.kind == "REVIEW":
            next_item_component, phase = act.component, "instruction"
            continue
        if act.kind in ("ADVANCE", "REMEDIATE"):
            c = act.component
            preface = teach(router, teacher, content.graph()[c].name, cfg.teach_token_cap)
        next_item_component, phase = c, "instruction"
    for comp in content.components():
        items_to_mastery.setdefault(comp.id, None)     # censored at cap / non-mastery
    # post-session assessment: instructed families (acquisition) and held-out families (transfer)
    post, transfer = [], []
    for comp in content.components():
        for _ in range(n_assess):
            i = present(router, student, content.sample_item(comp.id, student.rng), "", cfg.answer_token_cap,
                        telemetry, base, "post", h, None)
            post.append(i.correct)
        if content.held_out_families(comp.id):
            for _ in range(n_assess):
                i = present(router, student, content.sample_item(comp.id, student.rng, held_out=True), "",
                            cfg.answer_token_cap, telemetry, base, "transfer", h, None)
                transfer.append(i.correct)
    return {"outcome": outcome, "interactions": len(h.interactions), "mastered": sorted(policy.mastered),
            "items_to_mastery": items_to_mastery, "first_mastered": first_mastered,
            "post_accuracy": sum(post) / len(post) if post else None,
            "transfer_accuracy": sum(transfer) / len(transfer) if transfer else None,
            "history": h}
