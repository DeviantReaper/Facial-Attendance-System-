"""
face_utils.py — Core face processing logic
==========================================
Key improvements over the basic ChatGPT version:

1. get_face_encodings_robust()
   - Uses num_jitters=3 during registration for more stable embeddings.
   - Upsamples the image twice to catch smaller faces.

2. compare_faces_vote()
   - Each person can have MULTIPLE stored encodings (register same person several
     times with different angles/lighting).
   - Uses VOTING: the candidate who wins the most individual comparisons wins.
   - Falls back to average-distance ranking as a tiebreaker.
   - Two thresholds:
       STRICT  = 0.45  → used when a candidate wins a clear majority
       LOOSE   = 0.52  → used for single-encoding persons or ties (less confident)
"""

import face_recognition
import numpy as np
import cv2

# ── Tunable thresholds ──────────────────────────────────────────────────────
STRICT_THRESHOLD = 0.45   # high-confidence match
LOOSE_THRESHOLD  = 0.52   # acceptable match when only 1 encoding stored


def get_face_encodings_robust(image: np.ndarray) -> list:
    """
    Extract face encodings from a BGR image (as loaded by OpenCV).

    - Converts to RGB (face_recognition expects RGB).
    - Upsamples 2× to catch small/distant faces.
    - Uses num_jitters=3 for more stable embeddings at registration time.

    Returns a list of numpy arrays (one per face found).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=2, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations, num_jitters=3)
    return list(encodings)


def compare_faces_vote(name_encodings: dict[str, list], unknown_encoding: np.ndarray) -> str:
    """
    Identify an unknown face against all registered users.

    Parameters
    ----------
    name_encodings : dict  {name: [enc1, enc2, ...]}
    unknown_encoding : np.ndarray   128-d face embedding

    Returns
    -------
    str  — matched name, or "Unknown"
    """
    if not name_encodings:
        return "Unknown"

    vote_counts: dict[str, int]   = {}
    best_distances: dict[str, float] = {}

    for name, encodings in name_encodings.items():
        distances = face_recognition.face_distance(encodings, unknown_encoding)

        # Count how many stored encodings are within the strict threshold
        votes = int(np.sum(distances < STRICT_THRESHOLD))
        vote_counts[name] = votes
        best_distances[name] = float(np.min(distances))

    # ── Primary criterion: most votes ──────────────────────────────────────
    max_votes = max(vote_counts.values())

    if max_votes > 0:
        # Among candidates with the most votes, pick the closest distance
        top_candidates = [n for n, v in vote_counts.items() if v == max_votes]
        winner = min(top_candidates, key=lambda n: best_distances[n])

        if best_distances[winner] < STRICT_THRESHOLD:
            return winner

    # ── Fallback: minimum distance with loose threshold ────────────────────
    closest_name = min(best_distances, key=best_distances.get)
    if best_distances[closest_name] < LOOSE_THRESHOLD:
        return closest_name

    return "Unknown"
