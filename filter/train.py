# coding=utf-8
"""
阶段 1：用 labeled.csv 训练「明星 vs 非明星」二分类器。

用法（在 trendradar/filter 目录下）：
    uv run python train.py

产出：
    data/model.joblib   打包好的 (vectorizer, clf, threshold)
    data/metrics.txt    评估报告 + 错判样本，供人工复核
"""
import csv
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent / "data"
LABELED_CSV = DATA / "labeled_topics.csv"
MODEL_PATH = DATA / "model.joblib"
METRICS_PATH = DATA / "metrics.txt"

SEED = 42
random.seed(SEED)


def load_data():
    texts, labels = [], []
    with LABELED_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            texts.append(row["title"])
            labels.append(1 if row["keep"] == "true" else 0)
    return texts, labels


def build_pipeline():
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
    )
    return vectorizer, clf


def sweep_thresholds(y_true, proba):
    """按阈值扫描指标，返回最佳 F1 的阈值与表格文本。"""
    y_true = np.asarray(y_true)
    rows = []
    best_t, best_f1 = 0.5, -1.0
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
        rows.append(f"  阈值 {t:.1f}  precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}  "
                    f"(误杀 {fp} / 漏判 {fn})")
    return best_t, best_f1, rows


def main():
    texts, labels = load_data()
    print(f"样本 {len(texts)} 条：选题 {sum(labels)} / 噪音 {len(labels) - sum(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    vectorizer, clf = build_pipeline()
    Xtr = vectorizer.fit_transform(X_train)
    clf.fit(Xtr, y_train)

    Xte = vectorizer.transform(X_test)
    proba = clf.predict_proba(Xte)[:, 1]

    lines = []
    lines.append(f"样本总数 {len(texts)}（选题 {sum(labels)} / 噪音 {len(labels) - sum(labels)}）")
    lines.append(f"训练集 {len(X_train)} / 测试集 {len(X_test)}")
    lines.append(f"特征维度 {len(vectorizer.get_feature_names_out())}")
    lines.append(f"ROC AUC {roc_auc_score(y_test, proba):.4f}   "
                 f"PR AUC {average_precision_score(y_test, proba):.4f}   "
                 f"（基准：选题占比 {sum(y_test) / len(y_test):.3f}）\n")

    threshold, best_f1, table = sweep_thresholds(y_test, proba)
    lines.append("=== 阈值扫描（取 F1 最高）===")
    lines += table
    lines.append(f"  → 选定阈值 {threshold:.1f}（f1={best_f1:.3f}）\n")

    pred = (proba >= threshold).astype(int)
    lines.append("=== 分类报告（阈值 %.3f）===" % threshold)
    lines.append(classification_report(y_test, pred, target_names=["噪音", "选题"], digits=3))
    lines.append("=== 混淆矩阵 ===")
    lines.append(str(confusion_matrix(y_test, pred)))
    lines.append("  [[真·噪音, 误杀],\n   [漏判, 真·选题]]\n")

    fn = [X_test[i] for i in range(len(y_test)) if y_test[i] == 1 and pred[i] == 0]
    fp = [X_test[i] for i in range(len(y_test)) if y_test[i] == 0 and pred[i] == 1]
    lines.append(f"=== 漏判的选题（{len(fn)} 条，全列）===")
    lines += [f"  {t}" for t in sorted(fn)]
    lines.append(f"\n=== 误杀成噪音的（{len(fp)} 条，只列概率最高的 40 条）===")
    fp_scored = sorted(
        ((proba[i], X_test[i]) for i in range(len(y_test)) if y_test[i] == 0 and pred[i] == 1),
        reverse=True,
    )
    lines += [f"  {p:.3f}  {t}" for p, t in fp_scored[:40]]

    # 可解释性：权重最高的字符 n-gram
    names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    order = coef.argsort()
    lines.append("\n=== 最像选题的字符 n-gram（权重前 40）===")
    lines.append("  " + " | ".join(names[i] for i in order[-40:][::-1]))
    lines.append("\n=== 最像噪音的字符 n-gram（权重后 40）===")
    lines.append("  " + " | ".join(names[i] for i in order[:40]))

    # 用全量数据重训最终模型
    vectorizer_full, clf_full = build_pipeline()
    Xall = vectorizer_full.fit_transform(texts)
    clf_full.fit(Xall, labels)
    joblib.dump(
        {"vectorizer": vectorizer_full, "clf": clf_full, "threshold": threshold},
        MODEL_PATH,
    )

    report = "\n".join(lines)
    METRICS_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n模型已保存 → {MODEL_PATH}")


if __name__ == "__main__":
    main()
