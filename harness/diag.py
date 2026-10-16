"""Diagnostic: which part of the teacher prompt does a model refuse?
Sends several variants of the teacher turn and prints stop reason and text length for each.

  python -m harness.diag --model claude-fable-5-1
"""
import argparse, json
from anthropic import Anthropic
from .content import load_content
from .session import TEACHER_SYSTEM, TEACH_PROMPT, STUDENT_SYSTEM


def call(client, model, system, user, label):
    kw = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": user}]}
    if system:
        kw["system"] = system
    r = client.messages.create(**kw)
    text = "".join(getattr(b, "text", "") or "" for b in r.content)
    print(f"[{label}] stop_reason={r.stop_reason} out_tokens={r.usage.output_tokens} text={text[:80]!r}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--set", default="Y")
    a = ap.parse_args()
    c = Anthropic(max_retries=0)
    spec = load_content(a.set).specification()
    teach = TEACH_PROMPT.format(name="the pulse")
    variants = [
        ("1 full teacher prompt", TEACHER_SYSTEM + "\n\nWhat you know:\n" + spec, teach),
        ("2 no 'do not state answers' sentence", TEACHER_SYSTEM.replace("Do not state answers to specific test questions. ", "") + "\n\nWhat you know:\n" + spec, teach),
        ("3 no TEACH: prefix", TEACHER_SYSTEM + "\n\nWhat you know:\n" + spec, "Explain the component 'the pulse' to your student. Cover what it is, the rule, and one worked illustration."),
        ("4 spec only, no tutor role", "What you know:\n" + spec, "Explain the pulse rule to someone who has never seen it, with one worked example."),
        ("5 tutor role, no spec", TEACHER_SYSTEM, "Explain how compound interest works to a learner, with one worked example."),
        ("6 no system, plain request", None, "Here is a rule system.\n" + spec + "\nExplain the pulse rule to a learner with one worked example."),
        ("7 student item with 'leak'", STUDENT_SYSTEM, "A cairn stands at 22 marks with a stride of 9 and a leak of 6. After one pulse and its leak, what is its height?\n\nA. 27 marks\nB. 33 marks\nC. 28 marks\nD. 30 marks\n\nAnswer with your reasoning if you wish, then end with exactly one final line of the form\nANSWER: <letter>"),
        ("8 spec with 'DRAFT, not frozen' line removed", TEACHER_SYSTEM + "\n\nWhat you know:\n" + spec.replace("SPECIFICATION: cairn rule system (DRAFT, not frozen)\n", ""), teach),
    ]
    for label, system, user in variants:
        try:
            call(c, a.model, system, user, label)
        except Exception as e:
            print(f"[{label}] ERROR {e}")


if __name__ == "__main__":
    main()
