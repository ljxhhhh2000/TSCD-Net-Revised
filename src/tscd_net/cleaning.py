from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WEIBO_SHORT_URL_RE = re.compile(r"\bt\.cn/\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[^\s:：,，。！？!?]+")
_SPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+"
)


@dataclass
class TextCleaner:
    """Clean Weibo HTML without removing its semantic hashtag text."""

    remove_mentions: bool = True
    drop_punctuation: bool = False
    convert_traditional: bool = True

    def __post_init__(self) -> None:
        self._converter = None
        if self.convert_traditional:
            try:
                from opencc import OpenCC

                self._converter = OpenCC("t2s")
            except ImportError:
                # English and most posts are unaffected. The optional CJK extra
                # installs an OpenCC implementation when traditional conversion
                # is required for a particular run.
                self._converter = None

    def __call__(self, value: str | None) -> str:
        if not value:
            return ""

        text = unicodedata.normalize("NFKC", html.unescape(str(value)))
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        # Weibo emotion markup contains an image and an alt token. The paper
        # removes emotion characters, so the whole icon span is removed.
        for tag in soup.find_all(class_=lambda value: value and "url-icon" in value):
            tag.decompose()
        text = soup.get_text(" ", strip=True)

        text = _CONTROL_RE.sub(" ", text)
        text = _EMOJI_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        text = _WEIBO_SHORT_URL_RE.sub(" ", text)
        if self.remove_mentions:
            text = _MENTION_RE.sub(" ", text)
        if self._converter is not None:
            text = self._converter.convert(text)
        text = text.lower()

        # Keep Chinese, Latin text, digits and sentence punctuation. This is less
        # aggressive than discarding all non-word symbols and preserves the
        # sentence structure consumed by sentence-transformers.
        if self.drop_punctuation:
            text = re.sub(r"[^\w\u3400-\u9fff]+", " ", text)
        text = _SPACE_RE.sub(" ", text).strip()
        return text

