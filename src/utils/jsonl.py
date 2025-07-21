"""JSONL file utilities."""

import json
from pathlib import Path
from typing import Any, Iterator, List, Union


def read_jsonl(file_path: Union[str, Path]) -> Iterator[Any]:
    """Read JSONL file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(data: List[Any], file_path: Union[str, Path]) -> None:
    """Write data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(item: Any, file_path: Union[str, Path]) -> None:
    """Append single item to JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl_as_list(file_path: Union[str, Path]) -> List[Any]:
    """Load entire JSONL file as list."""
    return list(read_jsonl(file_path))