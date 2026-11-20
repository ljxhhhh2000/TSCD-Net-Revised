from datetime import datetime, timezone

from tscd_net.config import DataConfig
from tscd_net.data import _make_record
from tscd_net.symptoms import get_symptoms


def _payload(tweet_count: int, origin_every: int = 1) -> dict:
    tweets = []
    for index in range(tweet_count):
        minute = index % 60
        hour = index // 60
        tweets.append(
            {
                "text": f"post {index}",
                "post_time": (
                    f"Sun, 01 Jan 2023 {hour:02d}:{minute:02d}:00 +0000"
                ),
                "is_origin": index % origin_every == 0,
            }
        )
    return {
        "label": {"depressed": True, "symptoms": {"sadness": True}},
        "user": {"screen_name": "user"},
        "tweets": tweets,
    }


def test_minimum_is_applied_after_retweet_filtering() -> None:
    config = DataConfig(
        min_tweets=50,
        max_tweets=500,
        include_retweets=False,
    )
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    record = _make_record(
        _payload(60, origin_every=2),
        1,
        config,
        get_symptoms("paper15"),
        start,
        end,
        "depressed.jsonl",
        1,
    )
    assert record is None


def test_latest_window_is_retained() -> None:
    config = DataConfig(
        min_tweets=2,
        max_tweets=3,
        include_retweets=True,
    )
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    record = _make_record(
        _payload(6),
        1,
        config,
        get_symptoms("paper15"),
        start,
        end,
        "depressed.jsonl",
        1,
    )
    assert record is not None
    assert record.texts == ("post 3", "post 4", "post 5")
    assert record.eligible_tweet_count == 6
