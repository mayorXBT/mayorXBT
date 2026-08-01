from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets" / "source"
WIDTH, HEIGHT = 1180, 610
RNG = random.Random(42)


INFO_ROWS = [
    ("Subject", "Mayor / mayorXBT"),
    ("Role", "AI/NLP Engineer"),
    ("Origin", "Lagos, Nigeria"),
    ("Education", "BSc Computational Linguistics"),
    ("Status", "Building + Writing + Learning + Shipping"),
    ("ToolChain", "VS Code, Git, Codex, Figma"),
    ("Core.Lang", "Python, TypeScript, JavaScript"),
    ("Core.Frontend", "Next.js, React, Tailwind CSS"),
    ("Core.Backend", "FastAPI, Node.js, AI Agents"),
    ("Core.Database", "PostgreSQL"),
    ("Core.Infra", "Vercel, Railway, Docker, GitHub Actions"),
    ("Grid.Mail", "mayor@ghoste.trade"),
    ("Grid.Portfolio", "mayor.polyweb.pro"),
    ("Grid.Substack", "mayor.substack.com"),
    ("Grid.GitHub", "github.com/0xWeb3Mayor"),
]


PALETTES = {
    "dark": {
        "bg": "#070B14",
        "panel": "#0A101F",
        "panel2": "#0D1528",
        "border": "#24324C",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "dim": "#64748B",
        "chrome": "#22D3EE",
        "portrait": "#A78BFA",
        "accent": "#10B981",
    },
    "light": {
        "bg": "#EEF4FA",
        "panel": "#FFFFFF",
        "panel2": "#F8FAFC",
        "border": "#CBD5E1",
        "text": "#0F172A",
        "muted": "#475569",
        "dim": "#64748B",
        "chrome": "#0891B2",
        "portrait": "#7C3AED",
        "accent": "#059669",
    },
}


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def prepare_portrait(theme: str) -> np.ndarray:
    img = Image.open(SOURCE / "portrait.png").convert("RGB")
    size = min(img.size)
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size)).resize((300, 340))
    img = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return 1.0 - arr if theme == "light" else arr


def sample_portrait_points(theme: str, count: int = 3400) -> np.ndarray:
    density = prepare_portrait(theme)
    density = np.power(np.clip(density, 0.04, 1.0), 1.25)
    flat = density.ravel()
    probs = flat / flat.sum()
    rng = np.random.default_rng(412 if theme == "dark" else 824)
    ids = rng.choice(flat.size, size=count, replace=True, p=probs)
    ys, xs = np.divmod(ids, density.shape[1])
    xs = xs + rng.uniform(-0.38, 0.38, count)
    ys = ys + rng.uniform(-0.38, 0.38, count)
    return np.column_stack((76 + xs * 1.12, 145 + ys * 1.12))


def logo_mask_points(filename: str, count: int, seed: int) -> np.ndarray:
    img = Image.open(SOURCE / filename).convert("RGBA")
    bg = Image.new("RGBA", img.size, "white")
    bg.alpha_composite(img)
    gray = np.asarray(ImageOps.grayscale(bg.convert("RGB")).resize((220, 220)))
    if filename == "ethereum.png":
        mask = gray < 242
    else:
        mask = gray < 150
    coords = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    chosen = coords[rng.choice(len(coords), size=count, replace=True)]
    y, x = chosen[:, 0], chosen[:, 1]
    return np.column_stack((116 + x * 1.18, 198 + y * 1.18))


def build_svg(theme: str) -> str:
    p = PALETTES[theme]
    portrait = sample_portrait_points(theme)
    traveller_count = 500
    travellers = portrait[np.linspace(0, len(portrait) - 1, traveller_count).astype(int)]
    logos = [
        logo_mask_points("python.png", traveller_count, 10),
        logo_mask_points("openai.png", traveller_count, 20),
        logo_mask_points("ethereum.png", traveller_count, 30),
    ]

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "text{font-family:'SFMono-Regular',Consolas,'Liberation Mono',monospace}",
        ".blink{animation:blink 1.4s steps(2,end) infinite}.pulse{animation:pulse 1.8s ease-in-out infinite}",
        "@keyframes blink{50%{opacity:.25}}@keyframes pulse{50%{opacity:.45}}",
        "</style>",
        f'<rect width="1180" height="610" rx="22" fill="{p["bg"]}"/>',
        f'<rect x="18" y="18" width="1144" height="574" rx="16" fill="{p["panel"]}" stroke="{p["border"]}" stroke-width="2"/>',
        f'<rect x="18" y="18" width="1144" height="54" rx="16" fill="{p["panel2"]}"/>',
        f'<path d="M18 57V34Q18 18 34 18H1146Q1162 18 1162 34V72H18Z" fill="{p["panel2"]}"/>',
        '<circle cx="45" cy="45" r="6" fill="#FF5F57"/><circle cx="66" cy="45" r="6" fill="#FEBC2E"/><circle cx="87" cy="45" r="6" fill="#28C840"/>',
        f'<text x="590" y="50" text-anchor="middle" font-size="14" fill="{p["muted"]}">profile.sh --live</text>',
        f'<rect x="43" y="92" width="405" height="464" rx="12" fill="{p["panel2"]}" stroke="{p["border"]}"/>',
        f'<text x="65" y="121" font-size="13" font-weight="700" letter-spacing="1.4" fill="{p["chrome"]}">VISUAL.MAP</text>',
        f'<text x="420" y="121" text-anchor="end" font-size="11" fill="{p["dim"]}">MORPH.SEQ / 03</text>',
        f'<path d="M65 132H426" stroke="{p["border"]}"/>',
        '<g shape-rendering="crispEdges">',
    ]

    for i, (x, y) in enumerate(portrait):
        delay = (i % 61) * 0.018
        radius = 0.72 + (i % 5) * 0.055
        out.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{p["portrait"]}" opacity="0">'
            f'<animate attributeName="opacity" values="0;0.82;0.82;0;0;0;0.82" keyTimes="0;0.08;0.22;0.29;0.91;0.97;1" '
            f'dur="14.2s" begin="{delay:.3f}s" repeatCount="indefinite"/></circle>'
        )
    out.append("</g><g>")

    times = "0;0.22;0.30;0.43;0.51;0.64;0.72;0.85;0.93;1"
    for i, (start_x, start_y) in enumerate(travellers):
        pts = [travellers[i], logos[0][i], logos[0][i], logos[1][i], logos[1][i], logos[2][i], logos[2][i], travellers[i], travellers[i], travellers[i]]
        xs = ";".join(f"{v[0]:.2f}" for v in pts)
        ys = ";".join(f"{v[1]:.2f}" for v in pts)
        out.append(
            f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="1.25" fill="{p["chrome"]}" opacity="0">'
            f'<animate attributeName="cx" values="{xs}" keyTimes="{times}" dur="14.2s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{ys}" keyTimes="{times}" dur="14.2s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;0;0.95;0.95;0.95;0.95;0.95;0.95;0;0" keyTimes="{times}" dur="14.2s" repeatCount="indefinite"/>'
            "</circle>"
        )

    out.extend(
        [
            "</g>",
            f'<text x="245" y="535" text-anchor="middle" font-size="12" fill="{p["muted"]}">PYTHON  /  OPENAI  /  ETHEREUM</text>',
            f'<text x="486" y="112" font-size="13" font-weight="700" letter-spacing="1.4" fill="{p["chrome"]}">SYSTEM.INFO</text>',
            '<circle class="pulse" cx="1042" cy="107" r="5" fill="#FF5F57"/>',
            f'<text x="1054" y="112" font-size="12" font-weight="700" fill="{p["text"]}">LIVE</text>',
            f'<rect x="1087" y="91" width="51" height="24" rx="12" fill="{p["portrait"]}" opacity=".18"/>',
            f'<text x="1112.5" y="108" text-anchor="middle" font-size="11" font-weight="700" fill="{p["portrait"]}">XBT</text>',
            f'<path d="M486 126H1138" stroke="{p["border"]}"/>',
        ]
    )

    y = 154
    for idx, (label, value) in enumerate(INFO_ROWS):
        if idx in (6, 11):
            out.append(f'<path d="M486 {y - 10}H1138" stroke="{p["border"]}" stroke-dasharray="3 5"/>')
            y += 8
        label_end = 492 + len(label) * 8.0
        value_start = 1138 - len(value) * 8.0
        out.append(
            f'<text x="486" y="{y}" font-size="13.3" fill="{p["muted"]}">{xml_escape(label)}</text>'
            f'<path d="M{label_end:.1f} {y - 4}H{max(label_end + 14, value_start - 12):.1f}" '
            f'stroke="{p["dim"]}" stroke-width="1.4" stroke-dasharray="2 5"/>'
            f'<text x="1138" y="{y}" text-anchor="end" font-size="13.3" font-weight="600" '
            f'fill="{p["text"]}">{xml_escape(value)}</text>'
        )
        y += 25

    out.extend(
        [
            f'<rect x="486" y="543" width="652" height="1" fill="{p["border"]}"/>',
            f'<text x="486" y="570" font-size="12" fill="{p["accent"]}">$ mayor --build --write --ship</text>',
            f'<rect class="blink" x="785" y="559" width="8" height="14" fill="{p["chrome"]}"/>',
            "</svg>",
        ]
    )
    return "".join(out)


def main() -> None:
    for theme in ("dark", "light"):
        (ROOT / f"{theme}.svg").write_text(build_svg(theme), encoding="utf-8")
    print("Generated dark.svg and light.svg")


if __name__ == "__main__":
    main()
