import os
import io
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st
import torch

# Append Structure-CLIP path for import (absolute path inside project root)
PROJECT_ROOT = os.path.dirname(__file__)
THIRD_PARTY_PATH = os.path.join(PROJECT_ROOT, "third_party", "Structure-CLIP", "model")
if THIRD_PARTY_PATH not in os.sys.path:
    os.sys.path.insert(0, THIRD_PARTY_PATH)

import clip  # type: ignore
from transformers import pipeline


def safe_st_image(image: Image.Image, caption: str | None = None):
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        # for older Streamlit versions
        st.image(image, caption=caption, use_column_width=True)


def safe_st_graphviz(dot_source: str):
    try:
        st.graphviz_chart(dot_source, use_container_width=True)
    except TypeError:
        st.graphviz_chart(dot_source)


@st.cache_resource(show_spinner=False)
def load_clip_model(device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess, device


def compute_similarity(model, preprocess, device, image: Image.Image, texts: list[str]):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        logits_per_image, _, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    return probs


@st.cache_resource(show_spinner=False)
def load_owl_vit_pipeline(device: str):
    model_id = "google/owlvit-base-patch32"
    device_index = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    detector = pipeline(
        task="zero-shot-object-detection",
        model=model_id,
        device=device_index,
    )
    return detector


def draw_detection_boxes(image: Image.Image, detections: list[dict], score_threshold: float) -> Image.Image:
    from PIL import ImageDraw

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        score = float(det.get("score", 0))
        if score < score_threshold:
            continue
        label = str(det.get("label", ""))
        box = det.get("box", {})
        xmin, ymin = int(box.get("xmin", 0)), int(box.get("ymin", 0))
        xmax, ymax = int(box.get("xmax", 0)), int(box.get("ymax", 0))
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=3)
        text = f"{label} {score:.2f}"
        # 背景付きラベル
        text_w = draw.textlength(text)
        text_h = 14
        draw.rectangle([(xmin, max(0, ymin - text_h - 2)), (xmin + int(text_w) + 6, ymin)], fill=(0, 255, 0))
        draw.text((xmin + 3, max(0, ymin - text_h - 1)), text, fill=(0, 0, 0))
    return annotated


def parse_triples(input_text: str) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for line in input_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 支援フォーマット: "subject, relation, object" または "subject ->relation-> object"
        if "->" in line:
            parts = [p.strip() for p in line.split("->") if p.strip()]
            if len(parts) == 3:
                head, rel, tail = parts
                triples.append((head, rel, tail))
                continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            head, rel, tail = parts[0], parts[1], parts[2]
            triples.append((head, rel, tail))
    return triples


def build_graphviz_dot(triples: list[tuple[str, str, str]]) -> str:
    nodes: set[str] = set()
    edges: list[tuple[str, str, str]] = []
    for h, r, t in triples:
        if not h or not t:
            continue
        nodes.add(h)
        nodes.add(t)
        edges.append((h, t, r))
    lines = [
        "digraph SceneGraph {",
        "  rankdir=LR;",
        "  node [shape=ellipse, style=filled, color=lightgray, fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\"];",
    ]
    for n in sorted(nodes):
        safe = n.replace('"', '\"')
        lines.append(f'  "{safe}";')
    for h, t, r in edges:
        h_s = h.replace('"', '\"')
        t_s = t.replace('"', '\"')
        r_s = r.replace('"', '\"') if r else ""
        label = f' [label="{r_s}"]' if r_s else ""
        lines.append(f'  "{h_s}" -> "{t_s}"{label};')
    lines.append("}")
    return "\n".join(lines)


def crop_union(image: Image.Image, box_a: dict, box_b: dict, pad_ratio: float = 0.05) -> Image.Image:
    w, h = image.size
    x1 = min(int(box_a.get("xmin", 0)), int(box_b.get("xmin", 0)))
    y1 = min(int(box_a.get("ymin", 0)), int(box_b.get("ymin", 0)))
    x2 = max(int(box_a.get("xmax", 0)), int(box_b.get("xmax", 0)))
    y2 = max(int(box_a.get("ymax", 0)), int(box_b.get("ymax", 0)))
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return image.crop((x1, y1, x2, y2))


def infer_relations_with_clip(
    model,
    preprocess,
    device: str,
    image: Image.Image,
    detections: list[dict],
    relation_vocab: list[str],
    max_pairs: int = 10,
    prob_threshold: float = 0.35,
) -> list[tuple[str, str, str, float]]:
    # detections → (label, score, box) の上位候補
    items: list[tuple[str, float, dict]] = []
    for det in detections:
        try:
            items.append((str(det.get("label", "")), float(det.get("score", 0)), det.get("box", {})))
        except Exception:
            continue
    items.sort(key=lambda x: x[1], reverse=True)

    # 上位からペア列挙
    pairs: list[tuple[str, dict, str, dict, float]] = []
    n = len(items)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s_label, s_score, s_box = items[i]
            o_label, o_score, o_box = items[j]
            pairs.append((s_label, s_box, o_label, o_box, s_score * o_score))
    pairs.sort(key=lambda x: x[4], reverse=True)
    pairs = pairs[:max_pairs]

    results: list[tuple[str, str, str, float]] = []
    for subj_label, subj_box, obj_label, obj_box, _ in pairs:
        crop = crop_union(image, subj_box, obj_box)
        image_input = preprocess(crop).unsqueeze(0).to(device)
        texts = [f"a {subj_label} {rel} a {obj_label}" for rel in relation_vocab]
        if not texts:
            continue
        text_tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            logits_per_image, _, _ = model(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_rel = relation_vocab[best_idx]
        if best_prob >= prob_threshold:
            results.append((subj_label, best_rel, obj_label, best_prob))
    return results


def main():
    st.set_page_config(page_title="Structure-CLIP Demo", page_icon="🧩", layout="wide")
    st.title("Structure-CLIP 簡易デモ（Streamlit）")
    st.caption(
        "参考: zjukg/Structure-CLIP [GitHub]"
    )

    model, preprocess, device = load_clip_model()
    owl_vit = load_owl_vit_pipeline(device)

    with st.sidebar:
        st.header("入力")
        uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
        text_input = st.text_area(
            "テキスト（カンマ区切りで複数候補を入力可）",
            "a dog, a cat, a person, a car",
        )
        st.divider()
        sg_text = st.text_area(
            "構造トリプル (1行に1つ: 主語, 関係, 目的語 または 主語 ->関係-> 目的語)",
            "dog, on, couch\nperson ->holding-> phone",
            height=120,
        )
        detect_on = st.checkbox("テキストに基づくゼロショット検出 (OWL-ViT) を有効化", value=True)
        score_th = st.slider("検出スコアしきい値", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
        st.divider()
        relation_on = st.checkbox("検出結果から関係をCLIPで推定しシーングラフ自動生成", value=True)
        relation_vocab_text = st.text_input(
            "関係語彙（カンマ区切り）",
            "holding,on,next to,under,wearing,riding,behind,in front of,near,over",
        )
        relation_prob_th = st.slider("関係スコアしきい値", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        relation_max_pairs = st.number_input("関係推定で評価する上限ペア数", min_value=1, max_value=50, value=10)
        run = st.button("推論・可視化を実行")

    if uploaded is None:
        st.info("左のサイドバーから画像をアップロードしてください。")
        return

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    safe_st_image(image, caption="入力画像")

    texts = [t.strip() for t in text_input.split(",") if t.strip()]
    if not texts:
        st.warning("テキストが空です。")
        return

    if run:
        with st.spinner("モデル推論中..."):
            probs = compute_similarity(model, preprocess, device, image, texts)

        st.subheader("類似度（確率）")
        for label, p in sorted(zip(texts, probs), key=lambda x: x[1], reverse=True):
            st.write(f"{label}: {p:.3f}")

        st.caption(
            "注: ここでは CLIP による簡易テキスト-画像類似度を表示しています。本家 Structure-CLIP はシーングラフ知識を統合した拡張モデルです。"
        )

        det_results: list[dict] = []
        if detect_on:
            with st.spinner("OWL-ViT によるゼロショット物体検出中..."):
                det_results = owl_vit(image, candidate_labels=texts)
            st.subheader("テキスト条件の検出結果 (OWL-ViT)")
            annotated = draw_detection_boxes(image, det_results, score_threshold=score_th)
            safe_st_image(annotated, caption="検出とバウンディングボックス")
            with st.expander("検出詳細"):
                for det in det_results:
                    st.write({
                        "label": det.get("label"),
                        "score": float(det.get("score", 0)),
                        "box": det.get("box"),
                    })

        # 構造可視化（手動入力）
        triples = parse_triples(sg_text)
        if triples:
            st.subheader("シーングラフ（手動入力）")
            dot = build_graphviz_dot(triples)
            safe_st_graphviz(dot)

        # 構造可視化（自動推定）
        if relation_on and det_results:
            vocab = [v.strip() for v in relation_vocab_text.split(",") if v.strip()]
            with st.spinner("CLIP による関係推定中..."):
                rel_triples = infer_relations_with_clip(
                    model,
                    preprocess,
                    device,
                    image,
                    det_results,
                    vocab,
                    max_pairs=int(relation_max_pairs),
                    prob_threshold=float(relation_prob_th),
                )
            if rel_triples:
                st.subheader("シーングラフ（自動推定）")
                triples_with_scores = [(h, f"{r} ({s:.2f})", t) for (h, r, t, s) in rel_triples]
                dot_auto = build_graphviz_dot(triples_with_scores)
                safe_st_graphviz(dot_auto)
                with st.expander("推定関係の一覧"):
                    for h, r, t, s in rel_triples:
                        st.write({"subject": h, "relation": r, "object": t, "score": round(s, 3)})
            else:
                st.info("関係は検出しきい値を満たしませんでした。語彙/しきい値/ペア上限を調整してください。")


if __name__ == "__main__":
    main()


