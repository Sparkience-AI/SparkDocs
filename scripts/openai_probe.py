"""Check OpenAI connectivity and GPT-6 Luna access with one small SDK request."""

import argparse
import os
import socket
from pathlib import Path

import openai

HOST = "api.openai.com"
MODEL = "gpt-6-luna"


def read_key(path: Path) -> str:
    key = ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, _, value = line.strip().removeprefix("export ").partition("=")
        if name.strip() == "OPENAI_API_KEY":
            key = value.strip().strip('"').strip("'")
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Read only OPENAI_API_KEY from this file.")
    args = parser.parse_args()

    try:
        socket.getaddrinfo(HOST, 443)
        print(f"DNS resolved {HOST}.")
    except OSError:
        print(f"Direct DNS lookup failed for {HOST}. A configured HTTPS proxy may still work.")

    try:
        key = read_key(args.env_file) if args.env_file else os.environ.get("OPENAI_API_KEY", "")
    except (OSError, UnicodeError):
        print("Could not read the supplied env file.")
        return 2
    if not key.strip():
        print("Set OPENAI_API_KEY or pass --env-file to run the SDK check.")
        return 2

    try:
        with openai.OpenAI(
            api_key=key.strip(),
            base_url=f"https://{HOST}/v1",
            timeout=10.0,
            max_retries=0,
        ) as client:
            result = client.responses.create(
                model=MODEL,
                input="Reply with OK.",
                max_output_tokens=32,
                store=False,
            )
    except openai.APITimeoutError:
        print("The OpenAI SDK timed out.")
        return 1
    except openai.APIConnectionError:
        print("The OpenAI SDK could not connect. Check DNS, HTTPS connectivity, or the proxy.")
        return 1
    except openai.APIStatusError as error:
        print(f"OpenAI was reachable and returned HTTP {error.status_code}.")
        return 1

    print(f"OpenAI accepted the request for {MODEL}. Response status: {result.status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
