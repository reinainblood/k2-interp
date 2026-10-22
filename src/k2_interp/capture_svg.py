"""Portable SVG of exact selected-token routing weights (same values as the web heatmap)."""

from html import escape


def router_svg(view):
    layers = view["layers"]
    maximum = max((w for row in layers.values() for w in row.get("expert_weights", [])), default=1)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1030" height="460" viewBox="0 0 1030 460">',
        '<rect width="1030" height="460" fill="#101616"/>',
        '<g fill="#e0e9e3" font-family="monospace" font-size="11">',
        f'<text x="30" y="22">Router weights / p{view["position"]} / {escape(view["run_id"])}</text>',
        f'<text x="30" y="443">Logical expert ID 0:191 / layers 0:60 / max {maximum:.8f}</text></g>',
    ]
    for layer in range(61):
        row = layers.get(str(layer), {})
        weights = dict(zip(row.get("expert_ids", []), row.get("expert_weights", []), strict=True))
        if layer % 5 == 0:
            parts.append(
                f'<text x="8" y="{55 + layer * 6}" fill="#a4b9ad" font-family="monospace" font-size="9">{layer}</text>'
            )
        for expert in range(192):
            t = weights.get(expert, 0) / maximum if maximum else 0
            color = (
                "#1a2222"
                if not weights
                else f"rgb({int(40 + 122 * t)},{int(84 + 153 * t)},{int(66 + 109 * t)})"
            )
            parts.append(
                f'<rect x="{35 + expert * 5}" y="{50 + layer * 6}" width="4" height="5" fill="{color}"><title>L{layer} E{expert}: {weights.get(expert, 0) if weights else "uncollected"}</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts).encode()
