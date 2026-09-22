"""Theme injection and the small HTML components the pages share.

Every HTML helper returns a single-line string with no leading indentation,
because Streamlit renders markdown and four leading spaces would turn a
block into code.
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

BAND_COLOUR = {"ROUTINE": "#2fb47c", "MONITOR": "#f5a524", "PRIORITY": "#e5484d"}
TEAL, INDIGO, ROSE = "#0e9f8e", "#6c5ce7", "#f06292"
INK, MUTED = "#0f172a", "#64748b"

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="white", font=dict(family="Inter", color=INK)),
    colorway=[TEAL, INDIGO, ROSE, "#f5a524", "#2fb47c", "#38bdf8"],
)
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def inject_theme() -> None:
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _e(text) -> str:
    return html.escape(str(text))


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------- headings
def hero(eyebrow: str, title_html: str, lede: str) -> None:
    md(f'<div class="mch-hero"><div class="mch-eyebrow"><span class="dot"></span>{_e(eyebrow)}</div>'
       f'<h1>{title_html}</h1><p class="lede">{_e(lede)}</p></div>')


def page_title(title: str, subtitle: str = "", eyebrow: str | None = None) -> None:
    eb = (f'<div class="mch-eyebrow"><span class="dot"></span>{_e(eyebrow)}</div>'
          if eyebrow else "")
    md(f'<div class="mch-page-title">{eb}<h1>{_e(title)}</h1>'
       f'<p>{_e(subtitle)}</p></div>')


def section(title: str, caption: str = "") -> None:
    cap = f'<p style="color:{MUTED};margin:0 0 0.9rem 0;font-size:0.95rem">{_e(caption)}</p>' if caption else ""
    md(f'<h3 style="margin:1.4rem 0 0.25rem 0;font-size:1.25rem">{_e(title)}</h3>{cap}')


# ----------------------------------------------------------------- blocks
def stat_tiles(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="mch-stat"><div class="v">{_e(v)}</div>'
                    f'<div class="l">{_e(l)}</div></div>' for v, l in items)
    md(f'<div class="mch-stats">{cells}</div>')


def card(title: str, body: str, kicker: str | None = None, body_is_html=False) -> None:
    k = f'<div class="kicker">{_e(kicker)}</div>' if kicker else ""
    b = body if body_is_html else _e(body)
    md(f'<div class="mch-card">{k}<h4>{_e(title)}</h4><p>{b}</p></div>')


def callout(title: str, body: str) -> None:
    md(f'<div class="mch-callout"><h4>{_e(title)}</h4><p>{_e(body)}</p></div>')


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="mch-pill {tone}">{_e(text)}</span>'


def status_line(label: str, state: str = "ok") -> None:
    md(f'<div class="mch-status"><span class="led {state}"></span>{_e(label)}</div>')


def phase_header(name: str, sub: str) -> None:
    md(f'<div class="mch-phase"><div class="name">{_e(name)}</div>'
       f'<div class="sub">{_e(sub)}</div></div>')


def band_banner(band: str, n_flagged: int, n_adverse: int, description: str) -> None:
    md(f'<div class="mch-band {band}"><div><div class="t">{_e(band)}</div>'
       f'<div class="s">{_e(description)}</div></div>'
       f'<div class="n">{n_flagged}<small> / {n_adverse} adverse outcomes flagged</small></div></div>')


def outcome_card(o: dict) -> None:
    """One target: probability, operating threshold, national rate, and the
    conformal verdict, on a single bar."""
    p, thr, nat = o["probability"], o["threshold"], o["national_rate"]
    adverse = o["kind"] == "adverse"
    if adverse:
        status = pill("flagged", "flag") if o["flagged"] else pill("below threshold", "clear")
    else:
        status = pill("likely facility birth", "good") if o["flagged"] else pill("below threshold", "neutral")
    if o.get("committed") is False:
        status += " " + pill("model abstains at 90% coverage", "info")
    delta = o["delta_points"]
    delta_txt = f"{delta:+.1f} pt vs national"
    fill_cls = "fill adverse" if adverse else "fill"
    note = (f'<div class="mch-note">{_e(o["note"])}</div>' if o.get("note") else "")
    md(
        f'<div class="mch-outcome">'
        f'<div class="row"><div class="name">{_e(o["label"])}<code>{_e(o["target"])}</code></div>'
        f'<div class="prob">{p * 100:.1f}%<small>{_e(delta_txt)}</small></div></div>'
        f'<div class="mch-track"><div class="{fill_cls}" style="width:{max(1.5, p * 100):.1f}%"></div>'
        f'<div class="mark nat" style="left:{nat * 100:.1f}%" title="national rate"></div>'
        f'<div class="mark thr" style="left:{thr * 100:.1f}%" title="operating threshold"></div></div>'
        f'<div class="mch-legend"><span><i style="background:#0f172a"></i>threshold {thr * 100:.1f}%</span>'
        f'<span><i style="background:rgba(15,23,42,.35)"></i>national {nat * 100:.1f}%</span>'
        f'<span>lift ×{o["lift"]:.2f}</span><span>{_e(o["model"])}</span></div>'
        f'<div class="meta" style="margin-top:0.55rem">{status}</div>{note}'
        f'</div>')


def endpoint_row(method: str, path: str, desc: str) -> str:
    return (f'<div class="mch-endpoint"><span class="m {method}">{method}</span>'
            f'<div><div class="p">{_e(path)}</div><div class="d">{_e(desc)}</div></div></div>')


def footer() -> None:
    md('<div class="mch-footer">Population-level risk estimates from socio-demographic inputs · '
       'Not a diagnosis, not a substitute for clinical assessment · '
       'Data courtesy of The DHS Program and NIPORT, Bangladesh</div>')
