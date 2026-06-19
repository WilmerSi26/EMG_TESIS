"""Shared movement labels for EMG acquisition and training."""

from __future__ import annotations


FINGERS = ["pulgar", "indice", "medio", "anular", "menique"]
DIRECTIONS = ["cierre", "apertura"]

MOVEMENT_CLASSES = [
    "reposo",
    *(f"dedo_{finger}_{direction}" for finger in FINGERS for direction in DIRECTIONS),
    "cierre",
    "apertura",
    "pinza",
]

LABEL_ALIASES = {
    "reposo": "reposo",
    "reposo_inicial": "reposo",
    "reposo_final": "reposo",
    "pausa": "reposo",
    "descanso": "reposo",
    "relajacion": "reposo",
    "cierre": "cierre",
    "cierre_mano": "cierre",
    "cerrar": "cierre",
    "apertura": "apertura",
    "apertura_mano": "apertura",
    "abrir": "apertura",
    "pinza": "pinza",
    "pinza_fina": "pinza",
}

for _finger in FINGERS:
    for _direction in DIRECTIONS:
        _canonical = f"dedo_{_finger}_{_direction}"
        LABEL_ALIASES[_canonical] = _canonical
        LABEL_ALIASES[f"{_finger}_{_direction}"] = _canonical
        LABEL_ALIASES[f"{_direction}_{_finger}"] = _canonical
        LABEL_ALIASES[f"{_direction}_dedo_{_finger}"] = _canonical

# Old labels like "dedo_pulgar" are intentionally not mapped here because
# they do not tell whether the finger was opening or closing.

CLASS_COLORS = {
    "reposo": "#64748b",
    "dedo_pulgar_cierre": "#b91c1c",
    "dedo_pulgar_apertura": "#2563eb",
    "dedo_indice_cierre": "#ea580c",
    "dedo_indice_apertura": "#0891b2",
    "dedo_medio_cierre": "#ca8a04",
    "dedo_medio_apertura": "#4f46e5",
    "dedo_anular_cierre": "#9333ea",
    "dedo_anular_apertura": "#16a34a",
    "dedo_menique_cierre": "#db2777",
    "dedo_menique_apertura": "#0f766e",
    "cierre": "#dc2626",
    "apertura": "#2563eb",
    "pinza": "#7c3aed",
}


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    return LABEL_ALIASES.get(label.strip().lower())
