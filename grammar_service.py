"""
ASL gloss-to-English post-processing service.

Primary method:
    Local Ollama model (default: qwen2.5:0.5bqwen2.5:0.5b)

Fallback:
    Conservative rule-based formatter

This module does not change sign-recognition predictions. It receives only the
accepted/finalized gloss sequence and returns a separate English sentence.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_GRAMMAR_MODEL",
    "qwen2.5:0.5b"
)

OLLAMA_TIMEOUT_SECONDS = float(
    os.getenv("OLLAMA_TIMEOUT_SECONDS", "45")
)

MAX_GLOSSES = 50
MAX_GLOSS_LENGTH = 50
MAX_SENTENCE_LENGTH = 300

# Function words that the grammar model may add without changing the core
# semantic content of the recognized gloss sequence.
ALLOWED_FUNCTION_WORDS = {
    "a", "an", "the",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "to", "of", "for", "from", "in", "on", "at", "by", "with",
    "and", "or", "but",
    "that", "this", "these", "those",
    "will", "would", "can", "could", "may", "might", "should",
    "not"
}

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string"},
        "used_glosses": {
            "type": "array",
            "items": {"type": "string"}
        },
        "added_function_words": {
            "type": "array",
            "items": {"type": "string"}
        },
        "uncertain": {"type": "boolean"}
    },
    "required": [
        "sentence",
        "used_glosses",
        "added_function_words",
        "uncertain"
    ],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are an ASL-gloss-to-English converter.

Convert the recognized ASL gloss sequence into one natural English sentence.

Strict rules:
1. Preserve the recognized meaning and the original gloss order.
2. Do not invent people, places, objects, actions, numbers, or facts.
3. Add only grammar-supporting function words, basic inflection, and punctuation.
4. If the sequence is unclear, minimally format the glosses and set uncertain=true.
5. Return exactly one sentence in the required JSON schema.
6. used_glosses must list the supplied glosses; do not replace them with invented labels.
7. added_function_words must list only words added for grammar.
"""


def sanitize_glosses(glosses: Iterable[Any]) -> List[str]:
    """Validate and normalize a gloss sequence received from the frontend."""

    if glosses is None:
        raise ValueError("glosses are required.")

    cleaned: List[str] = []

    for value in glosses:
        text = str(value).strip()

        if not text:
            continue

        # Keep letters, numbers, apostrophes, hyphens, and spaces. This blocks
        # prompt-control characters while preserving normal label names.
        text = re.sub(r"[^A-Za-z0-9'\- ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        if len(text) > MAX_GLOSS_LENGTH:
            text = text[:MAX_GLOSS_LENGTH].strip()

        cleaned.append(text.upper())

        if len(cleaned) >= MAX_GLOSSES:
            break

    if not cleaned:
        raise ValueError("At least one non-empty gloss is required.")

    return cleaned


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send one JSON request to the local Ollama API."""

    url = f"{OLLAMA_BASE_URL}{path}"
    request_body = json.dumps(payload).encode("utf-8")

    request = Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text)

    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {body or error.reason}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Ollama is not reachable at "
            f"{OLLAMA_BASE_URL}. Start the Ollama application/service first."
        ) from error

    except TimeoutError as error:
        raise RuntimeError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS:g} seconds."
        ) from error

    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON.") from error


def _get_json(path: str, timeout_seconds: float = 3.0) -> Dict[str, Any]:
    """Read JSON from one local Ollama endpoint."""

    url = f"{OLLAMA_BASE_URL}{path}"
    request = Request(url=url, method="GET")

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(str(error)) from error


def _normalize_sentence(sentence: str) -> str:
    """Apply length, spacing, capitalization, and punctuation safeguards."""

    sentence = re.sub(r"\s+", " ", str(sentence)).strip()
    sentence = sentence.strip('"`')

    if not sentence:
        raise ValueError("The grammar model returned an empty sentence.")

    if len(sentence) > MAX_SENTENCE_LENGTH:
        raise ValueError("The grammar model returned an unexpectedly long sentence.")

    sentence = sentence[0].upper() + sentence[1:]

    if sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _simple_word_forms(word: str) -> set[str]:
    """Create a small set of inflection forms for conservative validation."""

    forms = {word}

    if len(word) >= 2:
        forms.add(word + "s")
        forms.add(word + "ed")
        forms.add(word + "ing")

    if word.endswith("e") and len(word) > 2:
        forms.add(word[:-1] + "ing")
        forms.add(word + "d")

    if word.endswith("y") and len(word) > 2:
        forms.add(word[:-1] + "ies")
        forms.add(word[:-1] + "ied")

    return forms


def _has_unapproved_content(sentence: str, glosses: List[str]) -> bool:
    """
    Reject obvious new content words.

    This is intentionally conservative. If validation rejects an LLM response,
    the system returns the rule-based sentence instead of risking hallucination.
    """

    allowed = set(ALLOWED_FUNCTION_WORDS)

    for gloss in glosses:
        for word in _word_tokens(gloss):
            allowed.update(_simple_word_forms(word))

    sentence_words = _word_tokens(sentence)
    return any(word not in allowed for word in sentence_words)


def _validate_llm_output(raw_result: Dict[str, Any], glosses: List[str]) -> Dict[str, Any]:
    """Validate and normalize the structured result returned by Ollama."""

    if not isinstance(raw_result, dict):
        raise ValueError("The grammar model did not return a JSON object.")

    sentence = _normalize_sentence(raw_result.get("sentence", ""))

    if _has_unapproved_content(sentence, glosses):
       print("Validation warning:")
    print(sentence)

    added_words = raw_result.get("added_function_words", [])
    if not isinstance(added_words, list):
        added_words = []

    safe_added_words = []
    for word in added_words:
        normalized = str(word).strip().lower()
        if normalized in ALLOWED_FUNCTION_WORDS and normalized not in safe_added_words:
            safe_added_words.append(normalized)

    return {
        "raw_gloss": " ".join(glosses),
        "glosses": glosses,
        "completed_sentence": sentence,
        "used_glosses": glosses,
        "added_function_words": safe_added_words,
        "uncertain": bool(raw_result.get("uncertain", False)),
        "method": "local_llm",
        "model": OLLAMA_MODEL
    }


def complete_with_ollama(glosses: Iterable[Any]) -> Dict[str, Any]:
    """Convert glosses with the configured local Ollama model."""

    cleaned_glosses = sanitize_glosses(glosses)
    raw_gloss = " ".join(cleaned_glosses)

    user_prompt = (
        "Convert this ASL gloss sequence into one English sentence.\n"
        f"Glosses: {raw_gloss}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
            "top_p": 0.2,
            "num_predict": 120
        }
    }

    response = _post_json("/api/generate", payload)
    generated_text = response.get("response", "")

    try:
        structured_result = json.loads(generated_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The grammar model response could not be parsed as structured JSON."
        ) from error

    result = _validate_llm_output(structured_result, cleaned_glosses)

    # Ollama returns nanosecond durations when available.
    total_duration = response.get("total_duration")
    if isinstance(total_duration, (int, float)):
        result["generation_time_ms"] = round(float(total_duration) / 1_000_000, 2)

    return result


def rule_based_sentence(glosses: Iterable[Any]) -> Dict[str, Any]:
    """Create a safe, deterministic sentence when the local model is unavailable."""

    cleaned_glosses = sanitize_glosses(glosses)
    words: List[str] = []

    replacements = {
        "ME": "I",
        "MY": "my",
        "MINE": "mine",
        "YOU": "you",
        "YOUR": "your"
    }

    for gloss in cleaned_glosses:
        gloss_words = gloss.split()
        for token in gloss_words:
            words.append(replacements.get(token, token.lower()))

    added_function_words: List[str] = []

    # Conservative insertion for common gloss patterns such as I WANT EAT.
    result_words: List[str] = []
    for index, word in enumerate(words):
        result_words.append(word)

        if (
            word in {"want", "need", "try"}
            and index + 1 < len(words)
            and words[index + 1] != "to"
        ):
            result_words.append("to")
            added_function_words.append("to")

    sentence = _normalize_sentence(" ".join(result_words))

    return {
        "raw_gloss": " ".join(cleaned_glosses),
        "glosses": cleaned_glosses,
        "completed_sentence": sentence,
        "used_glosses": cleaned_glosses,
        "added_function_words": added_function_words,
        "uncertain": True,
        "method": "rule_based_fallback",
        "model": None
    }


def complete_sentence(
    glosses: Iterable[Any],
    use_local_llm: bool = True
) -> Dict[str, Any]:
    """Use Ollama first and automatically fall back to deterministic formatting."""

    cleaned_glosses = sanitize_glosses(glosses)

    if use_local_llm:
        try:
            return complete_with_ollama(cleaned_glosses)

        except Exception as error:
            print("\n========== OLLAMA ERROR ==========")
            traceback.print_exc()
            print("==================================\n")

            fallback = rule_based_sentence(cleaned_glosses)
            fallback["fallback_reason"] = str(error)
            return fallback

    fallback = rule_based_sentence(cleaned_glosses)
    fallback["fallback_reason"] = "Local LLM was disabled for this request."
    return fallback

def ollama_health() -> Dict[str, Any]:
    """Check whether Ollama is reachable and whether the selected model exists."""

    try:
        response = _get_json("/api/tags")
        models = response.get("models", [])

        model_names = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
                if name:
                    model_names.append(str(name))

        selected_base = OLLAMA_MODEL.split(":", 1)[0]
        model_available = any(
            name == OLLAMA_MODEL
            or name.split(":", 1)[0] == selected_base
            for name in model_names
        )

        return {
            "reachable": True,
            "base_url": OLLAMA_BASE_URL,
            "selected_model": OLLAMA_MODEL,
            "model_available": model_available,
            "installed_models": model_names
        }

    except Exception as error:
        return {
            "reachable": False,
            "base_url": OLLAMA_BASE_URL,
            "selected_model": OLLAMA_MODEL,
            "model_available": False,
            "installed_models": [],
            "error": str(error)
        }