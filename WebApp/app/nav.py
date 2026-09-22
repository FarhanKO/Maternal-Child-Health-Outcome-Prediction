"""The page registry, so any view can link to another by its url_path."""
from __future__ import annotations

import streamlit as st

PAGES: dict[str, st.Page] = {}


def register(sections: dict[str, list[st.Page]]) -> None:
    for pages in sections.values():
        for page in pages:
            PAGES[page.url_path] = page


def page(url_path: str) -> st.Page:
    return PAGES[url_path]


def link(url_path: str, label: str, icon: str | None = None,
         width: str = "stretch") -> None:
    st.page_link(PAGES[url_path], label=label, icon=icon, width=width)
