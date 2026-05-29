from __future__ import annotations

NOT_FOUND = "The requested information is not available in the provided file."


def format_answer(source_result: str, explanation: str | None = None) -> str:
    source_result = source_result.strip() if source_result else NOT_FOUND
    if source_result == NOT_FOUND:
        explanation = "The provided file does not contain the requested information."
    elif not explanation:
        explanation = _default_explanation(source_result)
    return (
        "FINAL RESULT\n"
        f"{source_result}\n\n"
        "EXPLANATION\n"
        f"{explanation.strip()}"
    )


def format_table_answer(table_view: str, result: str | None = None) -> str:
    table_view = table_view.strip() if table_view else NOT_FOUND
    if table_view == NOT_FOUND:
        result = "The provided file does not contain the requested information."
    elif not result:
        result = "The table above is computed directly from the provided structured file."
    return (
        "TABLE VIEW\n"
        f"{table_view}\n\n"
        "RESULT\n"
        f"{result.strip()}"
    )


def format_json_answer(data: str, explanation: str | None = None) -> str:
    data = data.strip() if data else "{}"
    explanation = explanation or "The JSON block above was extracted from the provided file content."
    return (
        "EXTRACTED DATA\n"
        f"```json\n{data}\n```\n\n"
        "EXPLANATION\n"
        f"{explanation.strip()}"
    )


def _default_explanation(source_result: str) -> str:
    first_line = next((line.strip(" -") for line in source_result.splitlines() if line.strip()), source_result)
    return f"Based on the provided file, {first_line.rstrip('.')}. This answer uses only the extracted file content."
