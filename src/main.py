import argparse
from enum import Enum
from pathlib import Path
from typing import Iterator, List
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO, SAM
import time
from tcc_common.pass_detector import PassDetector
from tcc_common.team_classifier import TeamClassifier
from tcc_common.ball import BallTracker, BallAnnotator
from tcc_common.soccer import SoccerPitchConfiguration
from tcc_common.stats_exporter import save_stats_csv
import yaml

def load_config():
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config" / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> str:
    """Resolve caminhos do projeto independentemente do diretório atual."""
    return str(PROJECT_ROOT / path)


PLAYER_DETECTION_MODEL_PATH = project_path(config["paths"]["models"]["player"])
BALL_DETECTION_MODEL_PATH = project_path(config["paths"]["models"]["ball"])
PITCH_DETECTION_MODEL_PATH = project_path(config["paths"]["models"]["pitch"])
SAM_MODEL_PATH = project_path(config["paths"]["models"]["sam"])

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF0000', '#0000FF', '#000000', '#FFFF00', '#FFFFFF']
CUSTOM_PALETTE = sv.ColorPalette.from_hex(COLORS)

MASK_ANNOTATOR = sv.MaskAnnotator(
    color=CUSTOM_PALETTE,
    color_lookup=sv.ColorLookup.CLASS, # Usa o ID da classe para escolher a cor
    opacity=0.4 # Transparência (0.6 fica parecendo aquela "tinta" que fizemos antes)
)

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

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=CUSTOM_PALETTE,
    thickness=2
)

BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=CUSTOM_PALETTE,
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)

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


def validate_model_files(mode: Mode) -> None:
    """Falha cedo com uma instrução clara quando os pesos não estão presentes."""
    required_by_mode = {
        Mode.PLAYER_DETECTION: [PLAYER_DETECTION_MODEL_PATH],
        Mode.FULL_ANALYSIS: [
            PLAYER_DETECTION_MODEL_PATH,
            BALL_DETECTION_MODEL_PATH,
            SAM_MODEL_PATH,
        ],
        Mode.TEAM_CLASSIFICATION: [PLAYER_DETECTION_MODEL_PATH],
        Mode.BALL_DETECTION: [BALL_DETECTION_MODEL_PATH],
        Mode.PITCH_DETECTION: [PITCH_DETECTION_MODEL_PATH],
    }
    missing = [path for path in required_by_mode[mode] if not Path(path).is_file()]
    if missing:
        display_paths = []
        for path in missing:
            try:
                display_paths.append(str(Path(path).relative_to(PROJECT_ROOT)))
            except ValueError:
                display_paths.append(str(path))
        names = "\n  - ".join(display_paths)
        raise FileNotFoundError(
            "Modelos necessários não encontrados:\n"
            f"  - {names}\n"
            "Consulte a seção 'Modelos' do README.md."
        )


# Parâmetros
CONFIDENCE_THRESHOLD = config["detection"]["confidence_threshold"]
NMS_THRESHOLD = config["detection"]["nms_threshold"]
STRIDE = config["inference"]["stride"]

FRAMES_AMOSTRAGEM_CORES = 150
LIMITE_PROXIMIDADE_PASSE = 40
MIN_FRAMES_PROXIMOS_PASSE = 2
DURACAO_TEXTO_PASSE_FRAMES = 60

# --- Anotadores ---
box_annotator_time0 = sv.BoxAnnotator(color=sv.Color.RED)
box_annotator_time1 = sv.BoxAnnotator(color=sv.Color.BLUE)
label_annotator = sv.LabelAnnotator(color=sv.Color.WHITE, text_color=sv.Color.BLACK)
ball_annotator = sv.BoxAnnotator(color=sv.Color.GREEN)


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


# Substitua a função antiga por esta
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
    ball: sv.Detections,
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

    # +++++ COLE ESTE BLOCO NOVO +++++
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = (
            sv.Detections
            .from_ultralytics(result)
            .with_nms(threshold=NMS_THRESHOLD)
        )

        detections = tracker.update_with_detections(detections)

        # 1. Separa as detecções
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        
        # 2. Garante que os IDs dos árbitros sejam inteiros
        referees.class_id = referees.class_id.astype(int)

        # 3. Roda a classificação SÓ se houver jogadores (Corrige RuntimeWarning)
        if len(players) > 0:
            player_crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(
                player_crops,
                tracker_ids=players.tracker_id
            )
            players.class_id = players_team_id.astype(int)
		
            try:
                goalkeepers_team_id = resolve_goalkeepers_team_id(
                    players, players.class_id, goalkeepers)
            except TypeError:
                goalkeepers_team_id = np.zeros(len(goalkeepers), dtype=int)
            
            goalkeepers.class_id = goalkeepers_team_id.astype(int)
        else:
            goalkeepers.class_id = goalkeepers.class_id.astype(int)
        

        # 6. Junta todas as detecções (agora com .class_id corretos)
        detections = sv.Detections.merge([players, goalkeepers, referees])
        
        # 7. Cria os labels (rótulos)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        
        # 8. Chama os anotadores da forma moderna (sem 'custom_color_lookup')
        #    Eles agora usam o .class_id para colorir automaticamente.
        annotated_frame = MASK_ANNOTATOR.annotate(
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
        #overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
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

    # 1. Anotador de Linhas (ESQUELETO)
    # Aqui passamos a lista que criamos manualmente acima
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.WHITE, 
        thickness=2, 
        edges=EDGES_COMPLETOS 
    )

    # 2. Anotador de IDs (Para DEBUG)
    # Isso desenha o número em cima do ponto. Essencial para você corrigir a lista EDGES.
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

        # Verifica se detectou pontos antes de tentar desenhar
        if keypoints.xy.shape[1] > 0:
            
            # A. Desenha as linhas definidas
            annotated_frame = edge_annotator.annotate(
                scene=annotated_frame, 
                key_points=keypoints
            )
            
            # B. Desenha os números (IDs)
            # Mantenha isso ativado até você ter certeza que as linhas estão certas
            annotated_frame = vertex_annotator.annotate(
                annotated_frame, 
                keypoints, 
                CONFIG.labels if hasattr(CONFIG, 'labels') else None
            )

        yield annotated_frame

def get_ball_possession(
    player_detections: sv.Detections, 
    ball_detections: sv.Detections, 
    possession_threshold: float
) -> (int, int):
    """
    Verifica qual jogador está com a posse da bola.

    Retorna:
        (int, int): (tracker_id do jogador, class_id do time) ou (None, None)
    """
    if len(ball_detections) == 0 or len(player_detections) == 0:
        return None, None

    # Pega o centro da primeira bola detectada (assumindo uma bola)
    ball_center = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
    
    # Pega a âncora inferior dos jogadores (mais perto dos pés)
    player_anchors = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Calcula a distância de todos os jogadores para a bola
    distances = np.linalg.norm(player_anchors - ball_center, axis=1)
    
    # Encontra o jogador mais próximo
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]

    if min_distance < possession_threshold:
        # Encontramos o dono da bola!
        player_tracker_id = player_detections.tracker_id[min_distance_idx]
        player_team_id = player_detections.class_id[min_distance_idx]
        return player_tracker_id, player_team_id
        
    return None, None

# (A função 'get_crops' e 'resolve_goalkeepers_team_id' devem estar aqui em cima)

def run_full_analysis(source_video_path: str, device: str, target_video_path: str = "output.mp4") -> Iterator[np.ndarray]:
    """
    Executa a análise completa com SEGMENTAÇÃO COLORIDA POR TIME.
    """
    inicio = time.time()
    # --- FASE 1: Carregar Modelos ---
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    #pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    
    # Fase de fit com MobileSAM — roda uma vez, melhora qualidade do KMeans
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
    del sam_model  # libera memória GPU após o fit
    import torch; torch.cuda.empty_cache()
    print("Classificador de times treinado com SAM.")

    # --- FASE 2: Inicialização ---
    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    
    # Anotador específico para a bola (pintar de branco sólido)
    # Definimos a bola como uma "classe" nova no visualizador se quisermos usar mask, 
    # mas CircleAnnotator costuma ficar melhor pra bola. Vou manter o seu ou sugerir um Circle.
   # CORREÇÃO: Use DotAnnotator para "pontos" fixos
    ball_annotator = sv.DotAnnotator(radius=5, color=sv.Color.WHITE)
    # ... (Seu código de Slicer e VideoInfo continua igual) ...
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))
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
    frame_num = 0
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    print(f"Iniciando análise com SEGMENTAÇÃO ({fps} FPS)...")
    
    unique_ids = set()
    from collections import defaultdict
    tracking_frames = defaultdict(int)
    
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="collecting"):
        frame_num += 1
        annotated_frame = frame.copy()

        # === 1. Detecção do Campo ===
        #result_pitch = pitch_detection_model(frame, verbose=False)[0]
        #pitch_keypoints = sv.KeyPoints.from_ultralytics(result_pitch)

        # === 2. Detecção e Classificação de Jogadores ===
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
            players.class_id = players_team_id.astype(int)
            try:
                goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players.class_id, goalkeepers, sv.Detections.empty())
            except TypeError:
                goalkeepers_team_id = np.zeros(len(goalkeepers), dtype=int)
            
            goalkeepers.class_id = goalkeepers_team_id.astype(int)
        else:
            goalkeepers.class_id = goalkeepers.class_id.astype(int)
        
        # Junta todos (Jogadores, Goleiros, Juízes)
        all_detections = sv.Detections.merge([players, goalkeepers, referees])
        
        # === 3. AQUI ENTRA O SAM (SEGMENTAÇÃO) ===
        # Se houver detecções, passamos as caixas para o SAM gerar as máscaras
        #if len(all_detections) > 0:
            # O SAM usa as bounding boxes do YOLO como "prompt"
         #   sam_result = segmentation_model(frame, bboxes=all_detections.xyxy, verbose=False)[0]
            
            # Se o SAM retornou máscaras, injetamos elas no objeto 'detections' do Supervision
          #  if sam_result.masks is not None:
                # Converte máscaras para formato Bool e atribui ao Supervision
           #     all_detections.mask = sam_result.masks.data.cpu().numpy().astype(bool)

        # === 4. Bola ===
        ball_detections = slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)

        # === 5. Scout (Passes) ===
        pass_detector.update(frame_num, fps, all_detections, ball_detections)

        # === 6. Anotação ===
        
        # --- ANOTAÇÃO DE JOGADORES (SEGMENTADA) ---
        # Se tivermos máscaras, o MaskAnnotator vai pintar
        if all_detections.mask is not None:
            annotated_frame = MASK_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=all_detections
            )
        
        # Adiciona os números (IDs) em cima
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        annotated_frame = PLAYER_LABEL_ANNOTATOR.annotate(
            annotated_frame, all_detections, labels=labels)
        
        # Anota Bola
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        
        # Anota Passes
        annotated_frame = pass_detector.annotate_frame(annotated_frame, frame_num)
        
        yield annotated_frame
    fim = time.time()
    tempo_total = fim - inicio
    fps_processamento = total_frames / tempo_total
    print(f"FPS de processamento: {fps_processamento:.2f}")
    
    print(len(unique_ids))
    print(f"Jogadores únicos rastreados: {len(unique_ids)}")
    
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
        shots_list=getattr(pass_detector, "shots_list", []),
        possession_summary=pass_detector.possession_summary(),
        video_path=source_video_path,
        fps=fps,
    )
    print(f"Estatísticas salvas em: {csv_path}")
    
def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    validate_model_files(mode)
    source_path = Path(source_video_path).expanduser().resolve()
    target_path = Path(target_video_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Vídeo de entrada não encontrado: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
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

    video_info = sv.VideoInfo.from_video_path(str(source_path))
    written_frames = 0
    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)
            written_frames += 1

    if written_frames == 0 or not target_path.is_file() or target_path.stat().st_size == 0:
        raise RuntimeError("Nenhum frame foi gravado no vídeo anotado.")

    print(f"Vídeo anotado salvo em: {target_path} ({written_frames} frames)")

           # cv2.imshow("frame", frame)
           # if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
       # cv2.destroyAllWindows()


def cli() -> None:
    """Ponto de entrada da linha de comando do SPECTRA."""
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
