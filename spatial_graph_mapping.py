"""
거래 네트워크의 3D 그래프 임베딩 기반 가시성/차단 지수 산출
+ 실제 라벨 대비 통계적 유효성 검증 포함
"""
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import os
import sys

np.random.seed(42)

# ---------- 0. 입력 데이터 검증 ----------
def load_and_validate_data():
    if os.path.exists("graph_transactions.csv"):
        df = pd.read_csv("graph_transactions.csv")
        required_cols = {"sender", "receiver", "amount"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"경고: 필수 컬럼 누락 {missing}. 샘플 데이터로 대체합니다.")
            return generate_sample_data()
        print(f"기존 거래 데이터 로드: {len(df)}건")
        if "label" not in df.columns:
            df["label"] = 0
            print("주의: label 컬럼이 없어 검증 단계는 스킵됩니다.")
        return df
    else:
        print("graph_transactions.csv 없음 -> 샘플 데이터 생성")
        return generate_sample_data()

def generate_sample_data():
    rows = []
    for _ in range(3000):
        rows.append({'sender': np.random.randint(1000, 1200),
                     'receiver': np.random.randint(1000, 1200),
                     'amount': np.random.uniform(10, 5000), 'label': 0})
    for hub in range(2000, 2003):
        mules = np.random.randint(2100, 2130, 20)
        for m in mules:
            rows.append({'sender': int(m), 'receiver': hub,
                         'amount': np.random.uniform(8000, 9900), 'label': 1})
    return pd.DataFrame(rows)

df = load_and_validate_data()

# ---------- 1. 거래 네트워크 구축 ----------
G = nx.DiGraph()
node_label = {}
for _, row in df.iterrows():
    s, r = row['sender'], row['receiver']
    if G.has_edge(s, r):
        G[s][r]['weight'] += row['amount']
    else:
        G.add_edge(s, r, weight=row['amount'])
    if row.get('label', 0) == 1:
        node_label[s] = 1
        node_label[r] = 1

print(f"네트워크 규모: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")

if G.number_of_nodes() < 5:
    print("오류: 노드 수가 너무 적어 분석이 무의미합니다. 데이터를 확인하세요.")
    sys.exit(1)

# ---------- 2. 3D 공간 임베딩 ----------
print("3D 그래프 임베딩 계산 중 (force-directed layout)...")
pos_3d = nx.spring_layout(G, dim=3, weight='weight', seed=42, k=0.5, iterations=100)

# ---------- 3. 가시성 지수 (PageRank + 연결중심성) ----------
pagerank = nx.pagerank(G, weight='weight')
degree_centrality = nx.degree_centrality(G)
visibility_raw = {n: 0.5 * pagerank[n] + 0.5 * degree_centrality[n] for n in G.nodes()}
vmin, vmax = min(visibility_raw.values()), max(visibility_raw.values())
visibility_index = {n: (v - vmin) / (vmax - vmin + 1e-9) for n, v in visibility_raw.items()}

# ---------- 4. 차단율(occlusion) 지수 (커뮤니티 내부 고립도) ----------
G_undirected = G.to_undirected()
partition = community_louvain.best_partition(G_undirected, weight='weight')
community_sizes = pd.Series(partition).value_counts()

occlusion_index = {}
for node in G.nodes():
    comm_members = set(n for n, c in partition.items() if c == partition[node])
    if len(comm_members) <= 1:
        occlusion_index[node] = 0.0
        continue
    internal = sum(1 for _, v in G.out_edges(node) if v in comm_members) + \
               sum(1 for u, _ in G.in_edges(node) if u in comm_members)
    total = G.degree(node) or 1
    occlusion_index[node] = internal / total

def classify_spatial_property(node):
    occ = occlusion_index[node]
    comm_size = community_sizes[partition[node]]
    return "BLIND_SPOT" if (occ >= 0.8 and comm_size <= 60) else "OPEN_SPACE"

# ---------- 5. 최종 데이터프레임 ----------
records = []
for i, node in enumerate(G.nodes()):
    x, y, z = pos_3d[node]
    records.append({
        "point_id": f"P{i:05d}", "account_id": node,
        "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
        "visibility_index": round(visibility_index[node], 4),
        "occlusion_index": round(occlusion_index[node], 4),
        "spatial_property": classify_spatial_property(node),
        "community_id": partition[node],
        "community_size": int(community_sizes[partition[node]]),
        "actual_label": node_label.get(node, 0),
    })
spatial_df = pd.DataFrame(records)
spatial_df.to_csv("spatial_analysis_report.csv", index=False)
print(f"저장 완료: spatial_analysis_report.csv ({len(spatial_df)}개 계좌)")

# ---------- 6. 통계적 유효성 검증 (라벨이 있는 경우) ----------
has_real_labels = spatial_df["actual_label"].sum() > 0
if has_real_labels:
    from sklearn.metrics import classification_report, roc_auc_score
    spatial_df["predicted_flag"] = (spatial_df["spatial_property"] == "BLIND_SPOT").astype(int)
    print("\n=== BLIND_SPOT 분류의 실제 라벨 대비 성능 검증 ===")
    print(classification_report(spatial_df["actual_label"], spatial_df["predicted_flag"],
                                 target_names=["정상", "이상계좌"], zero_division=0))
    try:
        auc = roc_auc_score(spatial_df["actual_label"], spatial_df["occlusion_index"])
        print(f"occlusion_index 단독 ROC-AUC: {auc:.4f} "
              f"(0.5=무의미, 1.0=완벽 / 이 지표가 실제로 신호를 담고 있는지 확인하는 값)")
    except ValueError:
        auc = None
else:
    print("\n라벨 데이터가 없어 통계적 검증을 스킵합니다 (지표의 정성적 해석만 가능).")
    auc = None

# ---------- 7. 시각화 ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = spatial_df["spatial_property"].map({"OPEN_SPACE": "steelblue", "BLIND_SPOT": "crimson"})
ax.scatter(spatial_df["x"], spatial_df["y"], spatial_df["z"],
           c=colors, s=spatial_df["visibility_index"] * 200 + 10, alpha=0.7)
ax.set_title("거래 네트워크 3D 임베딩 (빨강=BLIND_SPOT, 크기=가시성)")
plt.tight_layout()
plt.savefig("spatial_aml_visualization.png", dpi=150)
print("시각화 저장: spatial_aml_visualization.png")

# ---------- 8. 방법론 문서 + README 자동 생성 (과장 없는 정확한 설명) ----------
auc_line = f"- occlusion_index 단독 ROC-AUC: **{auc:.4f}**\n" if auc else "- (라벨 없어 정량 검증 미실시)\n"
readme = f"""# Spatial Graph Mapping for AML Blind-Spot Detection

## 이 프로젝트가 실제로 하는 것 (정확한 설명)

이 프로젝트는 **물리적 3D 공간 스캔 기술(NeRF/DUSt3R)을 금융 데이터에 직접 적용한 것이 아닙니다.**
대신, 거래 네트워크(계좌=노드, 거래=엣지)를 force-directed 알고리즘으로 3차원 공간에
임베딩(embedding)하고, "가시성"과 "차단(사각지대)"이라는 물리적 공간 개념을
그래프 이론의 실제 지표로 재정의해 유비적으로 적용했습니다.

| 개념 | 실제 계산 방법 |
|---|---|
| 가시성 지수 (visibility_index) | PageRank(0.5) + 연결중심성(0.5) 가중합 |
| 차단율 지수 (occlusion_index) | 소속 커뮤니티 내부 거래 비율 (Louvain 커뮤니티 탐지 기반) |
| BLIND_SPOT 판정 | 차단율 ≥ 0.8 AND 커뮤니티 규모 ≤ 60 |

## 검증 결과
{auc_line}
자세한 수치는 `spatial_analysis_report.csv`, 시각화는 `spatial_aml_visualization.png`를 참고하세요.

## 왜 이 프레이밍이 중요한가
"3D 스캔 기술을 금융에 적용했다"는 설명은 근거가 약합니다(두 기술의 수학적 기반이 다름).
정확한 설명은 "그래프 임베딩 공간에서 물리적 직관을 재해석해 새로운 리스크 지표를
설계해본 실험"입니다. 이 프로젝트의 가치는 기술 이식이 아니라, **서로 다른 도메인의
개념을 정직하게 재정의하고 검증하는 사고 과정**에 있습니다.

## 실행 방법
```bash
pip install -r requirements.txt
python3 spatial_graph_mapping.py
```
"""
with open("README.md", "w") as f:
    f.write(readme)

with open("requirements.txt", "w") as f:
    f.write("pandas\nnumpy\nnetworkx\npython-louvain\nscikit-learn\nmatplotlib\n")

print("\nREADME.md, requirements.txt 자동 생성 완료. 바로 GitHub에 올리셔도 됩니다.")
