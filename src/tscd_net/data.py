from __future__ import annotations

import hashlib
import io
import json
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator, TextIO

from tqdm import tqdm

from .config import DataConfig, PROJECT_ROOT
from .symptoms import Symptom, get_symptoms, ground_truth_symptom_vector


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    screen_name: str
    label: int
    texts: tuple[str, ...]
    timestamps: tuple[datetime, ...]
    symptom_truth: tuple[int, ...]
    eligible_tweet_count: int
    source_file: str
    source_line: int


class TimestampParseError(ValueError):
    pass


def parse_weibo_time(value: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise TimestampParseError(value) from exc
    if parsed is None:
        raise TimestampParseError(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def iter_dataset(
    config: DataConfig,
    symptom_set: str = "paper15",
) -> Iterator[UserRecord]:
    """Yield eligible users in deterministic source-file order.

    Posts are sorted chronologically, filtered, and only then truncated to the
    most recent `max_tweets` records, matching the paper's preprocessing.
    The minimum-post threshold is also applied after filtering.
    """

    dataset_path = _resolve_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    symptoms = get_symptoms(symptom_set)
    start = _date_boundary(config.start_date)
    end = _date_boundary(config.end_date)
    emitted = {0: 0, 1: 0}
    seen = {0: 0, 1: 0}
    skipped = {"invalid": 0, "short": 0, "empty": 0}

    with ExitStack() as stack:
        streams = _open_streams(stack, dataset_path, config.members)
        for member, stream in zip(config.members, streams):
            if _class_complete(emitted, config):
                break
            for line_number, line in enumerate(stream, start=1):
                if (
                    config.scan_limit_per_file is not None
                    and line_number > config.scan_limit_per_file
                ):
                    break
                try:
                    payload = json.loads(line)
                    label = _extract_label(payload)
                    if (
                        config.max_users_per_class is not None
                        and emitted[label] >= config.max_users_per_class
                    ):
                        continue
                    record = _make_record(
                        payload=payload,
                        label=label,
                        config=config,
                        symptoms=symptoms,
                        start=start,
                        end=end,
                        member=member,
                        line_number=line_number,
                    )
                except ValueError:
                    skipped["invalid"] += 1
                    continue
                if record is None:
                    skipped["short"] += 1
                    continue
                if not record.texts:
                    skipped["empty"] += 1
                    continue
                seen[label] += 1
                emitted[label] += 1
                yield record

    print(
        "Dataset summary:",
        {
            "eligible": seen,
            "emitted": emitted,
            "skipped": skipped,
            "include_retweets": config.include_retweets,
            "window": "most_recent",
        },
    )


def count_eligible_users(
    config: DataConfig,
    symptom_set: str = "paper15",
) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for record in tqdm(
        iter_dataset(config, symptom_set),
        desc="Counting eligible users",
        unit="user",
    ):
        counts[record.label] += 1
    return counts


def _make_record(
    payload: dict,
    label: int,
    config: DataConfig,
    symptoms: tuple[Symptom, ...],
    start: datetime,
    end: datetime,
    member: str,
    line_number: int,
) -> UserRecord | None:
    tweets = payload.get("tweets")
    user = payload.get("user")
    label_payload = payload.get("label")
    if not isinstance(tweets, list) or not isinstance(user, dict):
        raise ValueError("Missing tweets or user")

    dated_texts: list[tuple[datetime, int, str]] = []
    for tweet_index, tweet in enumerate(tweets):
        if not isinstance(tweet, dict):
            continue
        text = tweet.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if not config.include_retweets and tweet.get("is_origin") is not True:
            continue
        try:
            posted_at = parse_weibo_time(tweet.get("post_time", ""))
        except TimestampParseError:
            continue
        if not start <= posted_at < end:
            continue
        dated_texts.append((posted_at, tweet_index, text.strip()))

    dated_texts.sort(key=lambda item: (item[0], item[1]))
    eligible_count = len(dated_texts)
    bounded = dated_texts[-config.max_tweets :]
    if len(bounded) < config.min_tweets:
        return None

    screen_name = str(user.get("screen_name") or "")
    stable_key = f"{member}:{line_number}:{screen_name}"
    user_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:20]
    symptom_payload = (
        label_payload.get("symptoms", {})
        if isinstance(label_payload, dict)
        else {}
    )
    return UserRecord(
        user_id=user_id,
        screen_name=screen_name,
        label=label,
        texts=tuple(item[2] for item in bounded),
        timestamps=tuple(item[0] for item in bounded),
        symptom_truth=tuple(
            ground_truth_symptom_vector(
                {"symptoms": symptom_payload},
                symptoms,
            )
        ),
        eligible_tweet_count=eligible_count,
        source_file=member,
        source_line=line_number,
    )


def _class_complete(emitted: dict[int, int], config: DataConfig) -> bool:
    limit = config.max_users_per_class
    return limit is not None and all(count >= limit for count in emitted.values())


def _extract_label(payload: dict) -> int:
    label = payload.get("label")
    if not isinstance(label, dict) or not isinstance(
        label.get("depressed"), bool
    ):
        raise ValueError("Missing boolean label.depressed")
    return int(label["depressed"])


def _open_member(archive: zipfile.ZipFile, name: str) -> TextIO:
    raw = archive.open(name, "r")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="strict")


def _open_streams(
    stack: ExitStack,
    dataset_path: Path,
    members: list[str],
) -> list[TextIO]:
    if dataset_path.is_dir():
        streams = []
        for member in members:
            path = dataset_path / member
            if not path.is_file():
                raise ValueError(f"Dataset directory is missing file: {path}")
            streams.append(
                stack.enter_context(path.open("r", encoding="utf-8"))
            )
        return streams
    if dataset_path.is_file() and zipfile.is_zipfile(dataset_path):
        archive = stack.enter_context(zipfile.ZipFile(dataset_path))
        available = set(archive.namelist())
        missing = set(members) - available
        if missing:
            raise ValueError(
                f"Dataset archive is missing members: {sorted(missing)}"
            )
        return [
            stack.enter_context(_open_member(archive, member))
            for member in members
        ]
    raise ValueError(
        "data.dataset_zip must be a zip archive or an extracted directory "
        "containing control.jsonl and depressed.jsonl"
    )


def _date_boundary(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _resolve_dataset_path(config: DataConfig) -> Path:
    path = Path(config.dataset_zip)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
