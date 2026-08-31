import argparse
import time
from enum import Enum
from typing import Iterator, List

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO, SAM

from tcc_common.pass_detector import PassDetector
from tcc_common.team_classifier import TeamClassifier
from tcc_common.ball import BallTracker, BallAnnotator, filter_ball_detections_on_pitch
from tcc_common.soccer import SoccerPitchConfiguration
from tcc_common.stats_exporter import save_stats_csv
from tcc_common.shot_detector import ShotDetector
from spectra.config import load_config

config = load_config()

PLAYER_DETECTION_MODEL_PATH = config["paths"]["models"]["player"]
BALL_DETECTION_MODEL_PATH = config["paths"]["models"]["ball"]
PITCH_DETECTION_MODEL_PATH = config["paths"]["models"]["pitch"]
SAM_MODEL_PATH = config["paths"]["models"]["sam"]

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF0000', '#0000FF', '#000000', '#FFFF00', '#FFFFFF']
CUSTOM_PALETTE = sv.ColorPalette.from_hex(COLORS)

# === LADO ESQUERDO (Time A) - 13 Pontos ===
# Supondo IDs de 0 a 12
EDGES_ESQUERDA = [
    (0, 1),   # Canto Sup -> Grande Area Sup (Linha de fundo)
    (1, 2),   # Grande Area Sup -> Pequena Area Sup (Linha de fundo)
    (2, 3),   # Pequena Area Sup -> Trave Sup
    (4, 5),   # Trave Inf -> Pequena Area Inf
    (5, 6),   # Pequena Area Inf -> Grande Area Inf
    (6, 7),   # Grande Area Inf -> Canto Inf
    
    (1, 8),   # Grande Area Sup (Fundo) -> Grande Area Sup (Frente)
    (8, 11),  # Grande Area Sup (Frente) -> Grande Area Inf (Frente)
    (11, 6),  # Grande Area Inf (Frente) -> Grande Area Inf (Fundo)
    
    (2, 9),   # Pequena Area Sup (Fundo) -> Pequena Area Sup (Frente)
    (9, 10),  # Pequena Area Sup (Frente) -> Pequena Area Inf (Frente)
    (10, 5)   # Pequena Area Inf (Frente) -> Pequena Area Inf (Fundo)
]

# === CENTRO - 6 Pontos ===
# Supondo IDs de 13 a 18
# Geralmente: 13(Top Line), 14(Bottom Line), 15(Center Spot), 16-18(Circle)
EDGES_CENTRO = [
    (13, 14), # Linha do Meio (Cima -> Baixo)
    (16, 17), (17, 18), (18, 16) # Triângulo ou contorno do círculo central (se houver mais pontos)
]

# === LADO DIREITO (Time B) - 13 Pontos ===
# Supondo IDs de 19 a 31 (Espelho do lado esquerdo)
EDGES_DIREITA = [
    (19, 20), (20, 21), (21, 22), # Linha de fundo cima
    (23, 24), (24, 25), (25, 26), # Linha de fundo baixo
    
    (20, 27), (27, 30), (30, 25), # Grande área
    (21, 28), (28, 29), (29, 24)  # Pequena área
]

# Une todas as listas
EDGES_COMPLETOS = EDGES_ESQUERDA + EDGES_CENTRO + EDGES_DIREITA

PLAYER_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=CUSTOM_PALETTE,
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
    color_lookup=sv.ColorLookup.CLASS,
)

# --- Enumeração de Modos ---
class Mode(Enum):
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    FULL_ANALYSIS = 'FULL_ANALYSIS'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    BALL_DETECTION = 'BALL_DETECTION'
    PITCH_DETECTION = 'PITCH_DETECTION'

    def __str__(self):
        return self.value


# Parâmetros
CONFIDENCE_THRESHOLD = config["detection"]["confidence_threshold"]
NMS_THRESHOLD = config["detection"]["nms_threshold"]
STRIDE = config["inference"]["stride"]

LIMITE_PROXIMIDADE_PASSE = 40
DURACAO_TEXTO_PASSE_FRAMES = 60


# --- Funções de Geração de Frames (Controladoras) ---

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """
    Resolve o ID do time dos goleiros com base na proximidade do centróide do time.
    Versão robusta que evita o RuntimeWarning.
    """
    # Se não houver goleiros, retorne um array vazio
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    # Se não houver jogadores para calcular os centróides, atribua um time padrão (ex: 0)
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Verifica se há jogadores em cada time antes de calcular a média
    team_0_players = players_xy[players_team_id == 0]
    team_1_players = players_xy[players_team_id == 1]

    # Se um time não tiver jogadores, use o centróide do outro time como fallback distante
    if len(team_0_players) == 0 and len(team_1_players) == 0:
        return np.zeros(len(goalkeepers), dtype=int) # Caso raro: sem jogadores
    
    if len(team_0_players) == 0:
        team_1_centroid = team_1_players.mean(axis=0)
        team_0_centroid = team_1_centroid + np.array([1e6, 1e6]) # Torna o time 0 muito distante
    elif len(team_1_players) == 0:
        team_0_centroid = team_0_players.mean(axis=0)
        team_1_centroid = team_0_centroid + np.array([1e6, 1e6]) # Torna o time 1 muito distante
    else:
        team_0_centroid = team_0_players.mean(axis=0)
        team_1_centroid = team_1_players.mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(goalkeepers_team_id, dtype=int)

def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    crops = []
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="collecting crops"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    box_annotator = sv.BoxAnnotator(color=CUSTOM_PALETTE, thickness=2)
    
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = (
            sv.Detections
            .from_ultralytics(result)
            .with_nms(threshold=NMS_THRESHOLD)
        )

        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        
        referees.class_id = referees.class_id.astype(int)

        if len(players) > 0:
            player_crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(
                player_crops,
                tracker_ids=players.tracker_id
            )
            players.class_id = players_team_id.astype(int)
		
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players.class_id, goalkeepers)
            
            goalkeepers.class_id = goalkeepers_team_id.astype(int)
        else:
            goalkeepers.class_id = goalkeepers.class_id.astype(int)
        

        detections = sv.Detections.merge([players, goalkeepers, referees])
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        
        annotated_frame = box_annotator.annotate(
            annotated_frame, detections)
        annotated_frame = PLAYER_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        
        yield annotated_frame

def run_player_detection(source_video_path: str, device: str, player_model: YOLO) -> Iterator[np.ndarray]:
    """
    Modo simples: Apenas detecta e rastreia jogadores.
    """
    tracker = sv.ByteTrack()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE)
    label_annotator_simple = sv.LabelAnnotator(color=sv.Color.WHITE, text_color=sv.Color.BLACK)

    for frame in frame_generator:
        result = player_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result).with_nms(threshold=NMS_THRESHOLD)
        detections = tracker.update_with_detections(detections)

        labels = [f"ID:{id_}" for id_ in detections.tracker_id]
        
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator_simple.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        yield annotated_frame

def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(128, 128),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = filter_ball_detections_on_pitch(frame, detections)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Desenha o campo segmentado e as linhas corretas.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.WHITE, 
        thickness=2, 
        edges=EDGES_COMPLETOS 
    )

    vertex_annotator = sv.VertexLabelAnnotator(
        color=sv.Color.from_hex('#FF1493'), # Rosa
        text_color=sv.Color.WHITE,
        border_radius=5,
        text_scale=0.5
    )

    print("Iniciando detecção de campo...")

    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()

        if keypoints.xy.shape[1] > 0:
            annotated_frame = edge_annotator.annotate(
                scene=annotated_frame, 
                key_points=keypoints
            )
            
            annotated_frame = vertex_annotator.annotate(
                annotated_frame, 
                keypoints, 
                CONFIG.labels if hasattr(CONFIG, 'labels') else None
            )

        yield annotated_frame

def run_full_analysis(source_video_path: str, device: str, target_video_path: str = "output.mp4") -> Iterator[np.ndarray]:
    """
    Executa a análise completa com SEGMENTAÇÃO COLORIDA POR TIME.
    """
    inicio = time.time()
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    print("Coletando frames para classificação de times (com MobileSAM)...")
    sam_model = SAM(SAM_MODEL_PATH)
    frame_generator_crops = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    
    fit_frames = []
    fit_detections = []
    for frame in tqdm(frame_generator_crops, desc='Coletando crops para fit'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        if len(players) > 0:
            fit_frames.append(frame)
            fit_detections.append(detections)

    if not fit_frames:
        print("Aviso: Nenhum frame com jogadores coletado.")
        return

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit_with_sam(
        frames=fit_frames,
        detections_list=fit_detections,
        sam_model=sam_model,
        player_class_id=PLAYER_CLASS_ID,
    )
    del sam_model
    import torch
    torch.cuda.empty_cache()
    print("Classificador de times treinado com SAM.")

    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = sv.DotAnnotator(radius=5, color=sv.Color.WHITE)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(128, 128),
    )
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = video_info.total_frames
    fps = video_info.fps
    
    pass_detector = PassDetector(
        proximity_limit=LIMITE_PROXIMIDADE_PASSE,
        min_frames=3,
        display_duration=DURACAO_TEXTO_PASSE_FRAMES,
        referee_class_id=REFEREE_CLASS_ID,
        possession_grace_frames=8,
        distance_factor=1.2
    )
    shot_config = config.get("analysis", {}).get("shot", {})
    shot_detector = ShotDetector(
        speed_threshold=float(shot_config.get("speed_threshold", 30.0)),
        proximity_limit=float(shot_config.get("proximity_limit", 140.0)),
        min_speed_ratio=float(shot_config.get("min_speed_ratio", 2.0)),
        confirmation_frames=int(shot_config.get("confirmation_frames", 3)),
        min_progress_px=float(shot_config.get("min_progress_px", 45.0)),
        direction_consistency=float(shot_config.get("direction_consistency", 0.8)),
        min_horizontal_ratio=float(shot_config.get("min_horizontal_ratio", 1.15)),
        display_duration=DURACAO_TEXTO_PASSE_FRAMES,
        referee_class_id=REFEREE_CLASS_ID,
        event_type="CANDIDATO_CHUTE",
    )
    frame_num = 0
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    print(f"Iniciando análise com SEGMENTAÇÃO ({fps} FPS)...")
    
    unique_ids = set()
    from collections import defaultdict
    tracking_frames = defaultdict(int)
    team_assignment_counts = defaultdict(int)
    
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="collecting"):
        frame_num += 1
        annotated_frame = frame.copy()

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = player_tracker.update_with_detections(detections)
        
    

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        referees.class_id = referees.class_id.astype(int)
        
        if players.tracker_id is not None:
            for tid in players.tracker_id:
                if tid is not None:
                    unique_ids.add(int(tid))
                    tracking_frames[int(tid)] += 1

        if len(players) > 0:
            player_crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(
                player_crops,
                tracker_ids=players.tracker_id
            )
            for team_id in players_team_id:
                team_assignment_counts[int(team_id)] += 1
            players.class_id = players_team_id.astype(int)
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players.class_id, goalkeepers)
            
            goalkeepers.class_id = goalkeepers_team_id.astype(int)
        else:
            goalkeepers.class_id = goalkeepers.class_id.astype(int)
        
        all_detections = sv.Detections.merge([players, goalkeepers, referees])

        ball_detections = slicer(frame).with_nms(threshold=0.1)
        ball_detections = filter_ball_detections_on_pitch(frame, ball_detections)
        ball_detections = ball_tracker.update(ball_detections)

        pass_detector.update(frame_num, fps, all_detections, ball_detections)
        shot_detector.update(
            frame_num,
            fps,
            all_detections,
            ball_detections,
            frame_width=frame.shape[1],
        )

        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        annotated_frame = PLAYER_LABEL_ANNOTATOR.annotate(
            annotated_frame, all_detections, labels=labels)
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        annotated_frame = pass_detector.annotate_frame(annotated_frame, frame_num)
        annotated_frame = shot_detector.annotate_frame(annotated_frame, frame_num)
        
        yield annotated_frame
    fim = time.time()
    tempo_total = fim - inicio
    fps_processamento = total_frames / tempo_total
    print(f"FPS de processamento: {fps_processamento:.2f}")
    
    print(len(unique_ids))
    print(f"Jogadores únicos rastreados: {len(unique_ids)}")
    total_assignments = sum(team_assignment_counts.values())
    if total_assignments:
        pct_team_0 = 100.0 * team_assignment_counts[0] / total_assignments
        pct_team_1 = 100.0 * team_assignment_counts[1] / total_assignments
        print(
            "Distribuição das classificações: "
            f"Time 0={pct_team_0:.1f}% | Time 1={pct_team_1:.1f}%"
        )
        if max(pct_team_0, pct_team_1) >= 95.0:
            print(
                "Aviso: classificação de times muito desequilibrada; "
                "revise as cores e os fallbacks antes de interpretar a posse."
            )
    
    tempos = [
    f / video_info.fps
    for f in tracking_frames.values()
    if f >= 25      # pelo menos 1 segundo
    ]
    print("Tempo médio:", np.mean(tempos))
    print("Tempo máximo:", np.max(tempos))
    print("Tempo mínimo:", np.min(tempos))
    
    percentuais = [
    f / video_info.total_frames * 100
    for f in tracking_frames.values()
    if f >= 25
    ]

    print(np.mean(percentuais))
    
    print("Processamento concluído.")
    csv_path = target_video_path.replace(".mp4", "_stats.csv")
    save_stats_csv(
        output_path=csv_path,
        passes_list=pass_detector.passes_list,
        shots_list=shot_detector.shots_list,
        possession_summary=pass_detector.possession_summary(),
        video_path=source_video_path,
        fps=fps,
    )
    print(f"Estatísticas salvas em: {csv_path}")

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.PLAYER_DETECTION:
        player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device, player_model=player_model)
    elif mode == Mode.FULL_ANALYSIS:
        frame_generator = run_full_analysis(
            source_video_path=source_video_path, device=device, target_video_path=target_video_path)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)


def cli() -> None:
    """Executa a interface de linha de comando do pipeline."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )


if __name__ == '__main__':
    cli()
