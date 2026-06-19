"""Movement labels for the final functional EMG control protocol."""

from __future__ import annotations


FINAL_MOVEMENT_CLASSES = [
    "reposo",
    "apertura",
    "cierre",
    "pulgar_apertura",
    "pulgar_cierre",
    "indice_medio_apertura",
    "indice_medio_cierre",
    "anular_menique_apertura",
    "anular_menique_cierre",
    "pinza",
]

LABEL_ALIASES = {
    "reposo": "reposo",
    "reposo_inicial": "reposo",
    "reposo_final": "reposo",
    "pausa": "reposo",
    "descanso": "reposo",
    "relajacion": "reposo",
    "apertura": "apertura",
    "apertura_mano": "apertura",
    "abrir": "apertura",
    "cierre": "cierre",
    "cierre_mano": "cierre",
    "cerrar": "cierre",
    "pulgar_apertura": "pulgar_apertura",
    "dedo_pulgar_apertura": "pulgar_apertura",
    "apertura_pulgar": "pulgar_apertura",
    "pulgar_cierre": "pulgar_cierre",
    "dedo_pulgar_cierre": "pulgar_cierre",
    "cierre_pulgar": "pulgar_cierre",
    "indice_medio_apertura": "indice_medio_apertura",
    "dedo_indice_apertura": "indice_medio_apertura",
    "dedo_medio_apertura": "indice_medio_apertura",
    "indice_apertura": "indice_medio_apertura",
    "medio_apertura": "indice_medio_apertura",
    "apertura_indice": "indice_medio_apertura",
    "apertura_medio": "indice_medio_apertura",
    "indice_medio_cierre": "indice_medio_cierre",
    "dedo_indice_cierre": "indice_medio_cierre",
    "dedo_medio_cierre": "indice_medio_cierre",
    "indice_cierre": "indice_medio_cierre",
    "medio_cierre": "indice_medio_cierre",
    "cierre_indice": "indice_medio_cierre",
    "cierre_medio": "indice_medio_cierre",
    "anular_menique_apertura": "anular_menique_apertura",
    "dedo_anular_apertura": "anular_menique_apertura",
    "dedo_menique_apertura": "anular_menique_apertura",
    "anular_apertura": "anular_menique_apertura",
    "menique_apertura": "anular_menique_apertura",
    "apertura_anular": "anular_menique_apertura",
    "apertura_menique": "anular_menique_apertura",
    "anular_menique_cierre": "anular_menique_cierre",
    "dedo_anular_cierre": "anular_menique_cierre",
    "dedo_menique_cierre": "anular_menique_cierre",
    "anular_cierre": "anular_menique_cierre",
    "menique_cierre": "anular_menique_cierre",
    "cierre_anular": "anular_menique_cierre",
    "cierre_menique": "anular_menique_cierre",
    "pinza": "pinza",
    "pinza_fina": "pinza",
}

CLASS_COLORS = {
    "reposo": "#64748b",
    "apertura": "#2563eb",
    "cierre": "#dc2626",
    "pulgar_apertura": "#0891b2",
    "pulgar_cierre": "#b91c1c",
    "indice_medio_apertura": "#16a34a",
    "indice_medio_cierre": "#ea580c",
    "anular_menique_apertura": "#4f46e5",
    "anular_menique_cierre": "#9333ea",
    "pinza": "#7c3aed",
}


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    return LABEL_ALIASES.get(label.strip().lower())

