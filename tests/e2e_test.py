"""
Simple end-to-end test harness

Usage:
  1) Put images for registration in `tests/register/` named like `Name_1.jpg`, `Name_2.jpg` (the part before the first underscore is used as the person's name).
  2) Put images to test recognition in `tests/recognize/`.
  3) Activate the project's Python 3.11 venv (the one created earlier):
       source .venv311/bin/activate
  4) Run:
       python tests/e2e_test.py

The script will:
 - Load each image in `tests/register/`, extract a single face encoding and store it in the SQLite DB (same DB used by the app).
 - Then load each image in `tests/recognize/`, detect faces and print the matched name(s).

This runs against the project's local DB and uses the same face utilities so it verifies the real pipeline.
"""

import os
import argparse
import pickle
from typing import List

import cv2
import numpy as np

from database import SessionLocal, engine
from models import User, Base
from face_utils import get_face_encodings_robust, compare_faces_vote

Base.metadata.create_all(bind=engine)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for fn in sorted(os.listdir(folder)):
        _, ext = os.path.splitext(fn)
        if ext.lower() in IMAGE_EXTS:
            files.append(os.path.join(folder, fn))
    return files


def register_from_folder(folder: str):
    imgs = find_images(folder)
    if not imgs:
        print(f"No images found in {folder}")
        return

    db = SessionLocal()
    try:
        for img_path in imgs:
            fname = os.path.basename(img_path)
            name = fname.split("_")[0] if "_" in fname else os.path.splitext(fname)[0]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {img_path}, skipping")
                continue

            encs = get_face_encodings_robust(img)
            if not encs:
                print(f"No face detected in {fname}, skipping")
                continue
            if len(encs) > 1:
                print(f"Multiple faces ({len(encs)}) in {fname}, skipping")
                continue

            encoding = encs[0]
            existing = db.query(User).filter(User.name == name).first()
            if existing:
                stored = pickle.loads(existing.encoding)
                stored.append(encoding)
                existing.encoding = pickle.dumps(stored)
                db.commit()
                print(f"Appended encoding for {name} from {fname} (total {len(stored)})")
            else:
                user = User(name=name, encoding=pickle.dumps([encoding]))
                db.add(user)
                db.commit()
                print(f"Registered {name} from {fname}")
    finally:
        db.close()


def recognize_from_folder(folder: str):
    imgs = find_images(folder)
    if not imgs:
        print(f"No images found in {folder}")
        return

    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("No registered users in DB. Run registration first.")
            return

        name_encodings = {u.name: pickle.loads(u.encoding) for u in users}
    finally:
        db.close()

    for img_path in imgs:
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping")
            continue

        encs = get_face_encodings_robust(img)
        if not encs:
            print(f"No faces detected in {fname}")
            continue

        results = []
        for enc in encs:
            name = compare_faces_vote(name_encodings, enc)
            results.append(name)

        print(f"{fname}: detected {len(results)} face(s) -> {results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register-dir', default='tests/register', help='Folder with images to register')
    parser.add_argument('--recognize-dir', default='tests/recognize', help='Folder with images to run recognition')
    args = parser.parse_args()

    print('--- REGISTER PHASE ---')
    register_from_folder(args.register_dir)
    print('\n--- RECOGNITION PHASE ---')
    recognize_from_folder(args.recognize_dir)
