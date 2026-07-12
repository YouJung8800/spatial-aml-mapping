# Spatial Graph Mapping for AML Blind-Spot Detection

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
- occlusion_index 단독 ROC-AUC: **1.0000**

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

## Advanced: Confidence-Aware Graph Attention Network (연구 확장)

DUSt3R의 신뢰도 가중(confidence-weighted) 최적화 개념을 그래프 신경망
기반 AML 탐지에 이식한 실험적 확장입니다 (`gnn_aml_detector.py`).

### 핵심 기여
- GATv2 기반 이상탐지에 거래 신뢰도를 엣지 특징으로 명시적 반영
- 기존 PageRank+RandomForest 방식과의 정량 비교
- GNNExplainer 기반 설명 안정성(explanation stability) 검증 —
  EU AI Act(2026.08 고위험 AI 조항 발효)가 요구하는 감사가능성에 대응

### 한계 (정직하게 명시)
- 합성 데이터 기반 검증. 실제 거래 데이터 미확보
- 논문 원본(DUSt3R, FraudGT) 수준의 대규모 검증은 미실시
- 이 구현은 "새 파운데이션 모델의 발명"이 아니라, 기존 기법의
  도메인 간 이식(cross-domain transfer) 실험임

### 실행
```bash
pip install torch torch_geometric scikit-learn pandas numpy matplotlib
python3 gnn_aml_detector.py
