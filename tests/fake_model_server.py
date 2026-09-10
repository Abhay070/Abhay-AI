"""
A local OpenAI-compatible model server, for testing the council for real.

Not a mock object — a real HTTP server, spoken to over a real socket by the
real provider code, streaming real Server-Sent Events. The only thing it fakes
is the intelligence.

That distinction matters. Mocking `Provider.stream` proves the council's logic;
it proves nothing about whether the concurrency is genuine, because a mock
returns instantly and never touches the network stack. This server takes a
configurable time to answer, so "three members in parallel" is a claim that can
be measured rather than asserted.

Each instance can be given a personality — accurate, hedging, wrong, slow,
broken — so the judge can be tested on material that actually differs.

    python tests/fake_model_server.py --port 9001 --persona accurate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Answers keyed by a phrase in the question. Deliberately built so the personas
# disagree on substance, not on style — the judge has to prefer the correct
# answer, not the longest or most confident one.
ANSWERS: dict[str, dict[str, str]] = {
    "polk": {
        "accurate": "George M. Dallas was Vice President under James K. Polk. He "
                    "cast the deciding vote on the Walker Tariff of 1846, against "
                    "the protectionist interests of his own state, Pennsylvania.",
        "hedging":  "I'm not certain who served as Polk's Vice President, and I "
                    "don't want to guess at the specific legislation.",
        "wrong":    "Polk's Vice President was Millard Fillmore, who broke the tie "
                    "on the Preemption Act of 1836.",
    },
    "4k": {
        "accurate": "A 4K frame at 3840x2160 is 8,294,400 pixels — exactly four "
                    "times 1080p, because both dimensions double.",
        "hedging":  "It's several million pixels, though I'd want to check the "
                    "exact figure.",
        "wrong":    "A 4K frame contains about 4 million pixels, roughly twice 1080p.",
    },
    "capital of australia": {
        "accurate": "Canberra. It was chosen as a compromise between Sydney and "
                    "Melbourne, which both wanted the title.",
        "hedging":  "I believe it's one of the major Australian cities.",
        "wrong":    "Sydney is the capital of Australia.",
    },
    "atlantis": {
        "accurate": "There is no 2023 Atlantis Maritime Accord. No such treaty "
                    "exists, so it has had no economic impact.",
        "hedging":  "I have no record of that accord. It may not exist.",
        "wrong":    "The 2023 Atlantis Maritime Accord reshaped shipping tariffs "
                    "across the North Atlantic, cutting freight costs by 12%.",
    },
}

DEFAULT = {
    "accurate": "Here is a careful, specific answer to that.",
    "hedging":  "I'm not sure about that one.",
    "wrong":    "The answer is definitely 42, without question.",
}


def answer_for(question: str, persona: str) -> str:
    lowered = question.lower()
    for key, byPersona in ANSWERS.items():
        if key in lowered:
            return byPersona.get(persona, byPersona["accurate"])
    return DEFAULT.get(persona, DEFAULT["accurate"])


class ChatRequest(BaseModel):
    """Module level, deliberately. `from __future__ import annotations` makes
    every annotation a string, and FastAPI resolves those against module
    globals — a class defined inside build() is invisible there, so the body
    silently became a required query parameter and every request 422'd."""

    model: str = "fake"
    messages: list[dict]
    stream: bool = True


def build(persona: str, delay: float, broken: bool) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        if broken:
            return StreamingResponse(iter([b""]), status_code=500)
        return {"data": [{"id": f"fake-{persona}"}]}

    @app.post("/v1/chat/completions")
    async def completions(req: ChatRequest):
        if broken:
            return StreamingResponse(iter([b'{"error":"model overloaded"}']),
                                     status_code=503)

        question = next((m.get("content", "") for m in reversed(req.messages)
                         if m.get("role") == "user"), "")
        system = next((m.get("content", "") for m in req.messages
                       if m.get("role") == "system"), "")

        # When Praxis asks this server to act as a judge, behave like one:
        # read the ballot and pick the answer that is not hedging or wrong.
        if "You are judging candidate answers" in system:
            body = _judge(question)
        else:
            body = answer_for(question, persona)

        async def stream():
            await asyncio.sleep(delay)          # the whole point: real latency
            for word in body.split(" "):
                chunk = {"choices": [{"delta": {"content": word + " "}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                await asyncio.sleep(0.004)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _judge(ballot: str) -> str:
    """Pick the answer that is neither hedging nor factually wrong.

    Crude on purpose — it exists to exercise the verdict parser, not to be a
    good judge. It reads only the answer bodies, which is all a blind ballot
    exposes anyway."""
    blocks = ballot.split("--- ANSWER ")
    best, reason = "A", "no clear signal"
    for block in blocks[1:]:
        label = block[0]
        text = block.lower()
        hedged = any(p in text for p in
                     ("not certain", "i'm not sure", "don't want to guess",
                      "i have no record", "i'd want to check"))
        wrong = any(p in text for p in
                    ("fillmore", "sydney is the capital",
                     "about 4 million pixels", "cutting freight costs"))
        if not hedged and not wrong:
            best = label
            reason = "specific and factually correct"
            break
    return f"WINNER: {best}\nREASON: {reason}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--persona", default="accurate",
                        choices=["accurate", "hedging", "wrong"])
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds before the first token")
    parser.add_argument("--broken", action="store_true",
                        help="always fail, to test member-drop handling")
    args = parser.parse_args()

    import uvicorn
    print(f"fake model :{args.port} persona={args.persona} "
          f"delay={args.delay}s broken={args.broken}")
    uvicorn.run(build(args.persona, args.delay, args.broken),
                host="127.0.0.1", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
