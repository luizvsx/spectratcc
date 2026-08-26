import numpy as np

from tcc_common.team_classifier import TeamClassifier
from tests.fixtures.images import solid_hsv_crop


def fitted_classifier(memory_frames=3):
    low_hue = solid_hsv_crop(10)
    high_hue = solid_hsv_crop(80)
    classifier = TeamClassifier(memory_frames=memory_frames)
    classifier.fit([low_hue, low_hue, high_hue, high_hue])
    return classifier, low_hue, high_hue


def test_fit_learns_threshold_and_separates_distinct_hues():
    classifier, low_hue, high_hue = fitted_classifier()

    predictions = classifier.predict([low_hue, high_hue])

    assert classifier.fitted is True
    np.testing.assert_array_equal(predictions, [0, 1])
    assert 10 < classifier._h_threshold < 80


def test_predict_before_fit_or_with_no_crops_is_empty():
    classifier = TeamClassifier()

    assert classifier.predict([solid_hsv_crop(10)]).size == 0
    classifier.fitted = True
    assert classifier.predict([]).size == 0


def test_empty_fit_does_not_claim_classifier_is_fitted():
    classifier = TeamClassifier()

    classifier.fit([])

    assert classifier.fitted is False


def test_tracker_memory_is_created_and_absent_tracks_expire():
    classifier, low_hue, _ = fitted_classifier(memory_frames=2)
    classifier.predict([low_hue], tracker_ids=np.array([42]))
    assert classifier._memory[42] == (0, 2)

    classifier.predict([low_hue], tracker_ids=np.array([7]))
    assert classifier._memory[42] == (0, 1)

    classifier.predict([low_hue], tracker_ids=np.array([7]))
    assert 42 not in classifier._memory


def test_masked_hue_requires_minimum_pixels():
    classifier = TeamClassifier(min_mask_pixels=5)
    crop = solid_hsv_crop(25)
    small_mask = np.zeros(crop.shape[:2], dtype=bool)
    small_mask[0, :4] = True
    valid_mask = np.ones(crop.shape[:2], dtype=bool)

    assert classifier._h_from_masked_crop(crop, small_mask) is None
    assert classifier._h_from_masked_crop(crop, valid_mask) == 25


def test_threshold_update_ignores_poorly_separated_samples():
    classifier, _, _ = fitted_classifier()
    original = classifier._h_threshold
    similar = [solid_hsv_crop(20)] * 4 + [solid_hsv_crop(24)] * 4

    classifier.update_threshold(similar)

    assert classifier._h_threshold == original
