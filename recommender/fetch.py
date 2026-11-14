"""Fetch a job posting URL safely and extract its readable text.

Guards against server-side request forgery: only http(s), no private, loopback or link-local
addresses (checked on every redirect hop), a size cap and a timeout.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx


class FetchError(ValueError):
    pass


def _check_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError("Only http and https links are supported.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise FetchError("That website could not be found.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise FetchError("That address is not allowed.")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def fetch_job_text(url: str, timeout: float = 8.0, max_bytes: int = 2_000_000) -> str:
    with httpx.Client(timeout=timeout, follow_redirects=False, headers={"User-Agent": "assessment-recommender/2.0"}) as client:
        for _ in range(4):
            _check_host(url)
            with client.stream("GET", url) as response:
                if response.is_redirect and response.headers.get("location"):
                    url = urljoin(url, response.headers["location"])
                    continue
                if response.status_code >= 400:
                    raise FetchError(f"The page returned an error ({response.status_code}).")
                body = b""
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > max_bytes:
                        raise FetchError("The page is too large.")
                content_type = response.headers.get("content-type", "")
                text = body.decode(response.encoding or "utf-8", errors="replace")
                text = html_to_text(text) if "html" in content_type or "<html" in text[:500].lower() else text
                if len(text) < 200:
                    raise FetchError("Couldn't find a job description on that page. Paste the text instead.")
                return text[:20000]
    raise FetchError("Too many redirects.")
