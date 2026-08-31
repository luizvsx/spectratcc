"""
Exportação de estatísticas do SPECTRA para CSV.
Cole em: tcc_common/stats_exporter.py
"""

import csv
import os
from typing import List, Dict
from datetime import datetime


def save_stats_csv(
    output_path: str,
    passes_list: List[Dict],
    shots_list: List[Dict],
    possession_summary: Dict,
    video_path: str = "",
    fps: float = 30.0,
) -> str:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Junta todos os eventos usando o tipo real do dicionário
    eventos = []

    for p in passes_list:
        eventos.append({
            "tipo":         p.get("tipo", "PASSE"),   # PASSE_CURTO ou PASSE_LONGO
            "frame":        p["frame"],
            "tempo_s":      p["tempo_s"],
            "time":         p["time"],
            "jogador_de":   p.get("de", ""),
            "jogador_para": p.get("para", ""),
            "distancia_px": p["distancia_px"],
            "velocidade_px": p.get("velocidade_px", ""),
            "confianca": p.get("confianca", ""),
            "frame_confirmacao": p.get("frame_confirmacao", ""),
        })

    for s in shots_list:
        eventos.append({
            "tipo":         s.get("tipo", "CHUTE"),
            "frame":        s["frame"],
            "tempo_s":      s["tempo_s"],
            "time":         s["time"],
            "jogador_de":   s.get("de", ""),
            "jogador_para": s.get("para", ""),
            "distancia_px": s["distancia_px"],
            "velocidade_px": s.get("velocidade_px", ""),
            "confianca": s.get("confianca", ""),
            "frame_confirmacao": s.get("frame_confirmacao", ""),
        })

    eventos.sort(key=lambda e: e["frame"])

    def filtrar(lst, tipo, time_str):
        return [e for e in lst if e.get("tipo") == tipo and time_str in e["time"]]

    def avg_dist(lst, tipo, time_str):
        vals = [e["distancia_px"] for e in lst if e.get("tipo") == tipo and time_str in e["time"]]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    todos = eventos  # alias para facilitar filtragem abaixo

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Cabeçalho
        writer.writerow(["# SPECTRA — Estatísticas de Jogo"])
        writer.writerow(["# Gerado em", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["# Vídeo", os.path.basename(video_path)])
        writer.writerow(["# FPS", fps])
        writer.writerow([])

        # Seção 1: Eventos detalhados
        writer.writerow(["## EVENTOS"])
        writer.writerow([
            "tipo", "frame", "tempo_s", "time", "jogador_de", "jogador_para",
            "distancia_px", "velocidade_px", "confianca", "frame_confirmacao",
        ])
        for ev in eventos:
            writer.writerow([
                ev["tipo"], ev["frame"], ev["tempo_s"], ev["time"],
                ev["jogador_de"], ev["jogador_para"], ev["distancia_px"],
                ev["velocidade_px"], ev["confianca"], ev["frame_confirmacao"],
            ])
        writer.writerow([])

        # Seção 2: Resumo
        pct0 = possession_summary.get("pct_time_0", 0.0)
        pct1 = possession_summary.get("pct_time_1", 0.0)
        frames_t0 = possession_summary.get("frames_time_0", 0)
        frames_t1 = possession_summary.get("frames_time_1", 0)
        frames_sem = possession_summary.get("frames_sem_posse", 0)
        total_frames = possession_summary.get("frames_total", 0)

        writer.writerow(["## RESUMO"])
        writer.writerow(["metrica", "time_0", "time_1", "total"])

        # Passes curtos
        pc0 = len(filtrar(todos, "PASSE_CURTO", "0"))
        pc1 = len(filtrar(todos, "PASSE_CURTO", "1"))
        writer.writerow(["passes_curtos", pc0, pc1, pc0 + pc1])

        # Passes longos
        pl0 = len(filtrar(todos, "PASSE_LONGO", "0"))
        pl1 = len(filtrar(todos, "PASSE_LONGO", "1"))
        writer.writerow(["passes_longos", pl0, pl1, pl0 + pl1])

        # Total de passes
        total_p0 = pc0 + pl0
        total_p1 = pc1 + pl1
        writer.writerow(["passes_total", total_p0, total_p1, total_p0 + total_p1])

        # Chutes
        ch0 = len(filtrar(todos, "CHUTE", "0"))
        ch1 = len(filtrar(todos, "CHUTE", "1"))
        writer.writerow(["chutes", ch0, ch1, ch0 + ch1])

        cand0 = len(filtrar(todos, "CANDIDATO_CHUTE", "0"))
        cand1 = len(filtrar(todos, "CANDIDATO_CHUTE", "1"))
        writer.writerow(["candidatos_chute", cand0, cand1, cand0 + cand1])

        # Posse
        writer.writerow(["posse_%", f"{pct0:.1f}", f"{pct1:.1f}", "100.0"])
        writer.writerow(["frames_posse", frames_t0, frames_t1, frames_t0 + frames_t1])
        writer.writerow(["frames_sem_posse", "", "", frames_sem])
        writer.writerow(["frames_total", "", "", total_frames])

        # Distâncias médias
        writer.writerow(["dist_media_passe_curto_px",
                         avg_dist(todos, "PASSE_CURTO", "0"),
                         avg_dist(todos, "PASSE_CURTO", "1"), ""])
        writer.writerow(["dist_media_passe_longo_px",
                         avg_dist(todos, "PASSE_LONGO", "0"),
                         avg_dist(todos, "PASSE_LONGO", "1"), ""])
        writer.writerow(["dist_media_chute_px",
                         avg_dist(todos, "CHUTE", "0"),
                         avg_dist(todos, "CHUTE", "1"), ""])

        if total_frames > 0 and fps > 0:
            writer.writerow(["duracao_video_s", "", "", round(total_frames / fps, 1)])

    return os.path.abspath(output_path)
