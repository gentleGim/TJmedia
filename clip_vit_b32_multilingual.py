"""run_clip_vqa.py – CLIP ViT-B-32 (Multilingual) VQA 5-Fold 실험 스크립트.

사용 모델:
  - 이미지 인코더 : clip-ViT-B-32  (sentence-transformers)
  - 텍스트 인코더 : clip-ViT-B-32-multilingual-v1  (한국어 지원)

접근 방식:
  1. 각 샘플에 대해 이미지 임베딩을 추출  (CLIP image encoder)
  2. 4개 선택지(a,b,c,d)에 대해 "질문: ... 정답 후보: ..." 텍스트 임베딩 추출
  3. 이미지-텍스트 코사인 유사도 → 4차원 벡터 (특징)
  4. fit(): LogisticRegression으로 학습
  5. predict_proba(): 학습된 분류기로 확률 예측
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── 프로젝트 모듈 ──
from team_experiment_template import ExperimentConfig, build_option_prompt, run_experiment

# ─────────────────────── 설정 ───────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image.MAX_IMAGE_PIXELS = None

# CLIP 모델 이름
CLIP_IMAGE_MODEL = "clip-ViT-B-32"
CLIP_TEXT_MODEL  = "clip-ViT-B-32-multilingual-v1"

TEXT_TEMPLATE_ID = "A"  # build_option_prompt 템플릿

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name()}")


# ─────────────────── CLIP 모델 로더 ─────────────────────
def load_clip_models():
    """sentence-transformers 기반 CLIP 이미지/텍스트 모델을 로드한다."""
    from sentence_transformers import SentenceTransformer

    print(f"[INFO] Loading CLIP image model: {CLIP_IMAGE_MODEL} ...")
    t0 = time.time()
    img_model = SentenceTransformer(CLIP_IMAGE_MODEL, device=DEVICE)
    print(f"[INFO]   loaded in {time.time()-t0:.1f}s")

    print(f"[INFO] Loading CLIP text model:  {CLIP_TEXT_MODEL} ...")
    t0 = time.time()
    txt_model = SentenceTransformer(CLIP_TEXT_MODEL, device=DEVICE)
    print(f"[INFO]   loaded in {time.time()-t0:.1f}s")

    return img_model, txt_model


# ─────────────────── 특징 추출기 ────────────────────────
def extract_clip_features(
    df: pd.DataFrame,
    img_model,
    txt_model,
    text_template_id: str = "A",
    batch_size: int = 64,
) -> np.ndarray:
    """이미지-텍스트 코사인 유사도 4차원 특징 벡터를 추출한다.

    Returns:
        features: (N, 4) ndarray  –  각 선택지와의 코사인 유사도
    """
    option_keys = ["a", "b", "c", "d"]
    n = len(df)
    features = np.zeros((n, 4), dtype=np.float32)

    # ── 이미지 임베딩 (배치) ──
    print(f"  [CLIP] Encoding {n} images ...")
    images = []
    for _, row in tqdm(df.iterrows(), total=n, desc="  Loading images", leave=False):
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        images.append(img)

    img_embeddings = img_model.encode(
        images,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # 정규화
    img_norms = np.linalg.norm(img_embeddings, axis=1, keepdims=True)
    img_norms = np.where(img_norms == 0, 1.0, img_norms)
    img_embeddings = img_embeddings / img_norms

    # 메모리 해제
    del images
    gc.collect()

    # ── 각 선택지별 텍스트 임베딩 및 유사도 ──
    for opt_idx, opt_key in enumerate(option_keys):
        print(f"  [CLIP] Encoding texts for option '{opt_key}' ...")
        texts = []
        for _, row in df.iterrows():
            prompt = build_option_prompt(row, opt_key, text_template_id)
            texts.append(prompt)

        txt_embeddings = txt_model.encode(
            texts,
            batch_size=batch_size * 2,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        # 정규화
        txt_norms = np.linalg.norm(txt_embeddings, axis=1, keepdims=True)
        txt_norms = np.where(txt_norms == 0, 1.0, txt_norms)
        txt_embeddings = txt_embeddings / txt_norms

        # 코사인 유사도 (내적) ── 이미 정규화되어 있으므로
        sims = np.sum(img_embeddings * txt_embeddings, axis=1)
        features[:, opt_idx] = sims

        del texts, txt_embeddings
        gc.collect()

    del img_embeddings
    gc.collect()

    return features


# ─────────────────── CLIPMultiRunner ────────────────────
class CLIPMultiRunner:
    """CLIP ViT-B-32 multilingual 기반 VQA 러너.

    FoldRunner 인터페이스:
      - fit(train_df, valid_df)
      - predict_proba(df) → (N, 4) ndarray
    """

    # 모델을 클래스 변수로 공유 (fold마다 다시 로드하지 않음)
    _img_model = None
    _txt_model = None

    def __init__(self, config: ExperimentConfig, fold: int) -> None:
        self.config = config
        self.fold = fold
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            random_state=config.seed,
        )

        # 모델 로드 (최초 1회)
        if CLIPMultiRunner._img_model is None:
            CLIPMultiRunner._img_model, CLIPMultiRunner._txt_model = load_clip_models()

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_clip_features(
            df,
            CLIPMultiRunner._img_model,
            CLIPMultiRunner._txt_model,
            text_template_id=self.config.text_template_id,
            batch_size=self.config.batch_size,
        )

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        print(f"\n{'='*60}")
        print(f"  Fold {self.fold}: Extracting TRAIN features ({len(train_df)} rows)")
        print(f"{'='*60}")
        X_train = self._extract(train_df)
        y_train = train_df["label_id"].to_numpy()

        print(f"\n  Fold {self.fold}: Fitting scaler + LogisticRegression ...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)

        # 학습 정확도
        train_acc = self.classifier.score(X_train_scaled, y_train)
        print(f"  Fold {self.fold}: Train accuracy = {train_acc:.4f}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        print(f"\n  Fold {self.fold}: Extracting features ({len(df)} rows) for prediction ...")
        X = self._extract(df)
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)
        return probs.astype(np.float32)


# ─────────────────── 메인 실행 ──────────────────────────
def main():
    config = ExperimentConfig(
        run_id="clip-vit-b32-multi-v1",
        model_kind="clip",
        model_name="clip-ViT-B-32-multilingual-v1",
        text_template_id="A",
        image_size=224,
        batch_size=64,
        lr=0.0,        # LogisticRegression (sklearn)
        epochs=0,       # CLIP feature-based
        seed=42,
        n_splits=5,
        active_folds=(0, 1, 2, 3, 4),
        output_dir="outputs",
        train_csv="train.csv",
        dev_csv="dev.csv",
        test_csv="test.csv",
        note_lines=[
            "CLIP ViT-B-32 image encoder + multilingual text encoder",
            "Features: cosine similarity (image, question+option) x 4 options",
            "Classifier: LogisticRegression(C=1.0)",
            "Text template: A (질문+정답후보)",
        ],
    )

    def make_runner(cfg, fold):
        return CLIPMultiRunner(cfg, fold)

    print("\n" + "=" * 70)
    print("  CLIP ViT-B-32 (Multilingual) VQA Experiment")
    print("=" * 70)

    summary = run_experiment(config, make_runner)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  OOF Accuracy : {summary['oof_accuracy']:.6f}")
    print(f"  Dev Accuracy : {summary['dev_accuracy']:.6f}")
    print(f"  Notes        : {summary['notes_path']}")
    print(f"  Submission   : {summary['submission_path']}")
    for fs in summary["fold_scores"]:
        print(f"    Fold {int(fs['fold'])}: valid={fs['valid_accuracy']:.6f}, dev={fs['dev_accuracy']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
