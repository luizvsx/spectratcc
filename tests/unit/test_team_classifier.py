"""
Testes para TeamClassifier refatorado.
Cole em: tests/unit/test_team_classifier.py
"""
import numpy as np
import pytest
import supervision as sv
from types import SimpleNamespace
from tcc_common.team_classifier import TeamClassifier


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_red_crop(n=1):
    """Crop com camiseta vermelha (BGR)."""
    crops = []
    for _ in range(n):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :] = [30, 30, 200]   # região superior = vermelho (BGR)
        img[32:, :] = [30, 120, 30]   # região inferior = verde (grama)
        crops.append(img)
    return crops


def make_blue_crop(n=1):
    """Crop com camiseta azul (BGR)."""
    crops = []
    for _ in range(n):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :] = [200, 30, 30]   # região superior = azul (BGR)
        img[32:, :] = [30, 120, 30]   # região inferior = verde (grama)
        crops.append(img)
    return crops


def make_noisy_crop(base_crop, noise_level=80):
    """Adiciona ruído a um crop — simula instabilidade de detecção."""
    noisy = base_crop.copy().astype(np.int16)
    noisy += np.random.randint(-noise_level, noise_level, noisy.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def fit_two_teams(clf, n=10):
    """Fita o classificador com n crops de cada time."""
    all_crops = make_red_crop(n) + make_blue_crop(n)
    clf.fit(all_crops)
    return clf


# ── Testes básicos ────────────────────────────────────────────────────────────

class TestBasic:

    def test_fit_sets_fitted_flag(self):
        clf = TeamClassifier()
        clf.fit(make_red_crop(5) + make_blue_crop(5))
        assert clf.fitted

    def test_fit_empty_does_not_crash(self):
        clf = TeamClassifier()
        clf.fit([])
        assert not clf.fitted

    def test_predict_before_fit_returns_empty(self):
        clf = TeamClassifier()
        result = clf.predict(make_red_crop(3))
        assert len(result) == 0

    def test_predict_empty_crops_returns_empty(self):
        clf = fit_two_teams(TeamClassifier())
        assert len(clf.predict([])) == 0

    def test_predict_returns_correct_length(self):
        clf = fit_two_teams(TeamClassifier())
        crops = make_red_crop(4) + make_blue_crop(3)
        result = clf.predict(crops)
        assert len(result) == 7

    def test_predict_only_valid_labels(self):
        clf = fit_two_teams(TeamClassifier())
        crops = make_red_crop(5) + make_blue_crop(5)
        result = clf.predict(crops)
        assert set(result).issubset({0, 1})


# ── Separação de times ────────────────────────────────────────────────────────

class TestTeamSeparation:

    def test_distinct_colors_classified_differently(self):
        """Vermelho e azul devem cair em clusters distintos."""
        clf = fit_two_teams(TeamClassifier(), n=15)

        red_preds  = clf.predict(make_red_crop(5))
        blue_preds = clf.predict(make_blue_crop(5))

        assert len(set(red_preds))  == 1, "vermelho deve ser consistente"
        assert len(set(blue_preds)) == 1, "azul deve ser consistente"
        assert red_preds[0] != blue_preds[0], "times devem ter labels diferentes"

    def test_labels_sum_to_both_teams(self):
        """Com dois times iguais, ambos os labels devem aparecer."""
        clf = fit_two_teams(TeamClassifier())
        preds = clf.predict(make_red_crop(5) + make_blue_crop(5))
        assert 0 in preds
        assert 1 in preds

    def test_anchor_stability_after_refit(self):
        """
        Refitar com mesmas cores não deve inverter os labels.
        Testa que a ancoragem é determinística.
        """
        clf = TeamClassifier()
        crops = make_red_crop(10) + make_blue_crop(10)

        clf.fit(crops)
        preds_before = clf.predict(make_red_crop(5))

        clf.fit(crops)  # refita com os mesmos dados
        preds_after = clf.predict(make_red_crop(5))

        # Vermelho deve receber o mesmo label nas duas rodadas
        assert set(preds_before) == set(preds_after)


# ── Memória por tracker_id ────────────────────────────────────────────────────

class TestMemory:

    def test_memory_created_after_confident_predict(self):
        """Predict confiante deve criar entrada na memória."""
        clf = fit_two_teams(TeamClassifier(memory_frames=10), n=20)
        clf.predict(make_red_crop(1), tracker_ids=np.array([5]))
        assert 5 in clf._memory

    def test_memory_label_used_when_player_absent(self):
        """
        Quando jogador some do frame (não aparece no predict), memória deve
        ser decrementada mas mantida por memory_frames frames.
        """
        clf = fit_two_teams(TeamClassifier(memory_frames=5), n=20)

        # Cria memória para jogador 5
        label_5 = clf.predict(make_red_crop(1), tracker_ids=np.array([5]))[0]
        assert 5 in clf._memory

        # Jogador 5 some — apenas jogador 99 aparece por 3 frames
        for _ in range(3):
            clf.predict(make_blue_crop(1), tracker_ids=np.array([99]))

        # Memória do 5 ainda deve existir (dentro da janela de 5 frames)
        assert 5 in clf._memory
        assert clf._memory[5][0] == label_5  # label preservado

    def test_memory_preserves_label_for_same_crop(self):
        """
        Mesmo crop repetido deve sempre retornar o mesmo label (sem flickering).
        """
        clf = fit_two_teams(TeamClassifier(memory_frames=15), n=20)
        red = make_red_crop(1)

        labels = [clf.predict(red, tracker_ids=np.array([7]))[0] for _ in range(10)]

        # Todos os labels devem ser iguais — sem flickering
        assert len(set(labels)) == 1, f"flickering detectado: {labels}"

    def test_memory_without_tracker_ids_does_not_crash(self):
        """Sem tracker_ids, o predict deve funcionar normalmente sem memória."""
        clf = fit_two_teams(TeamClassifier())
        result = clf.predict(make_red_crop(3))
        assert len(result) == 3

    def test_different_players_independent_memory(self):
        """Memória de um jogador não deve contaminar outro."""
        clf = fit_two_teams(TeamClassifier(memory_frames=10), n=20)

        red_crop  = make_red_crop(1)
        blue_crop = make_blue_crop(1)

        # Jogador 1 = vermelho, jogador 2 = azul
        label_1 = clf.predict(red_crop,  tracker_ids=np.array([1]))[0]
        label_2 = clf.predict(blue_crop, tracker_ids=np.array([2]))[0]

        # Labels devem ser diferentes
        assert label_1 != label_2

        # Verifica que memória separada foi criada
        assert 1 in clf._memory
        assert 2 in clf._memory
        assert clf._memory[1][0] != clf._memory[2][0]

    def test_memory_expires(self):
        """Após memory_frames sem aparecer, a memória deve ser removida."""
        clf = fit_two_teams(TeamClassifier(memory_frames=3), n=20)

        # Cria memória para jogador 7
        clf.predict(make_red_crop(1), tracker_ids=np.array([7]))
        assert 7 in clf._memory

        # Roda vários frames sem o jogador 7 aparecer
        for _ in range(5):
            clf.predict(make_blue_crop(1), tracker_ids=np.array([99]))

        # Memória de 7 deve ter expirado
        assert 7 not in clf._memory


# ── Extração de feature (região da camiseta) ─────────────────────────────────

class TestFeatureExtraction:

    def test_shirt_region_differs_from_full_crop(self):
        """
        Feature extraída só da camiseta deve ser diferente da feature
        extraída do crop inteiro (que inclui fundo e pernas).
        """
        # Crop com camiseta vermelha na metade superior e grama na inferior
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        crop[:32, :] = [30, 30, 200]   # camiseta vermelha
        crop[32:, :] = [30, 180, 30]   # grama verde

        clf_shirt = TeamClassifier(shirt_region=0.45)
        clf_full  = TeamClassifier(shirt_region=1.0)

        feat_shirt = clf_shirt._extract_feature(crop)
        feat_full  = clf_full._extract_feature(crop)

        assert not np.allclose(feat_shirt, feat_full, atol=5), \
            "feature da camiseta deve diferir do crop completo"

    def test_feature_is_3d_vector(self):
        clf = TeamClassifier()
        crop = make_red_crop(1)[0]
        feat = clf._extract_feature(crop)
        assert feat.shape == (3,)

    def test_similar_shirts_produce_similar_features(self):
        """Dois crops da mesma camiseta devem ter features próximas."""
        clf = TeamClassifier()
        crops = make_red_crop(2)
        f1 = clf._extract_feature(crops[0])
        f2 = clf._extract_feature(crops[1])
        assert np.linalg.norm(f1 - f2) < 10.0


class TestSamFit:

    def test_fit_with_sam_learns_two_teams(self):
        frames = []
        detections = []
        for color in ([30, 30, 200], [30, 30, 200],
                      [200, 30, 30], [200, 30, 30]):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :32] = color
            frames.append(frame)
            detections.append(sv.Detections(
                xyxy=np.array([[0, 0, 32, 64]], dtype=float),
                class_id=np.array([2]),
            ))

        class FakeTensor:
            def __init__(self, array):
                self.array = array

            def cpu(self):
                return self

            def numpy(self):
                return self.array

        class FakeSam:
            def __call__(self, frame, bboxes, verbose=False):
                masks = np.zeros((len(bboxes), 64, 64), dtype=bool)
                for index, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
                    masks[index, y1:y2, x1:x2] = True
                return [SimpleNamespace(
                    masks=SimpleNamespace(data=FakeTensor(masks))
                )]

        classifier = TeamClassifier()
        classifier.fit_with_sam(frames, detections, FakeSam())

        assert classifier.fitted
        red_team = classifier.predict(make_red_crop(1))[0]
        blue_team = classifier.predict(make_blue_crop(1))[0]
        assert red_team != blue_team
