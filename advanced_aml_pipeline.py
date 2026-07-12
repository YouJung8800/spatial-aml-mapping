import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Mac 환경 한글 폰트 및 시각화 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 비지도 학습 기반 신뢰도 생성기 (Autoencoder)
# ==========================================
class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

# ==========================================
# 2. 데이터 불균형 대응 Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

# ==========================================
# 3. 신뢰도 인지 GAT 모델 (Confidence-Aware GAT)
# ==========================================
class ConfidenceGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, edge_dim=1)
        self.gat2 = GATv2Conv(hidden_channels, out_channels, edge_dim=1)

    def forward(self, x, edge_index, edge_confidence):
        x = self.gat1(x, edge_index, edge_attr=edge_confidence)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_confidence)
        return x

# ==========================================
# 4. 실전 파이프라인 시뮬레이션
# ==========================================
def run_advanced_pipeline():
    print("🚀 [단계 1] 극도 불균형 금융 데이터 시뮬레이션 (정상 95%, 사기 5%)...")
    num_nodes = 1000
    num_features = 10
    x = torch.randn((num_nodes, num_features))
    
    # 5%만 자금세탁(AML) 노드로 설정
    labels = torch.zeros(num_nodes)
    aml_indices = torch.randperm(num_nodes)[:50]
    labels[aml_indices] = 1.0
    
    edge_index = torch.randint(0, num_nodes, (2, 5000))

    print("🧠 [단계 2] 비지도 학습(Autoencoder)으로 숨겨진 이상치(Anomaly Score) 탐지...")
    autoencoder = FeatureAutoencoder(num_features, 4)
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    
    # 정상 데이터로만 학습한다고 가정
    normal_x = x[labels == 0]
    for _ in range(50):
        optimizer_ae.zero_grad()
        loss = F.mse_loss(autoencoder(normal_x), normal_x)
        loss.backward()
        optimizer_ae.step()

    # 전체 데이터 복원 오차를 기반으로 신뢰도(Confidence) 부여
    with torch.no_grad():
        reconstructed = autoencoder(x)
        mse = torch.mean((x - reconstructed)**2, dim=1)
        # 오차가 크면 신뢰도 0, 작으면 신뢰도 1
        node_confidence = torch.exp(-mse) 
        
        # 노드 신뢰도를 엣지 신뢰도로 변환 (출발/도착 노드의 평균)
        edge_confidence = (node_confidence[edge_index[0]] + node_confidence[edge_index[1]]) / 2.0
        edge_confidence = edge_confidence.unsqueeze(1)

    print("⚙️ [단계 3] Focal Loss가 적용된 Confidence-Aware GAT 학습...")
    model = ConfidenceGAT(num_features, 16, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    focal_loss = FocalLoss(alpha=0.8, gamma=2.0) # 소수 클래스(AML)에 강력한 가중치

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, edge_confidence).squeeze()
        loss = focal_loss(out, labels)
        loss.backward()
        optimizer.step()

    print("📊 [단계 4] PR-AUC (Precision-Recall) 평가 및 시각화...")
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(x, edge_index, edge_confidence).squeeze()).numpy()
        y_true = labels.numpy()
        
    pr_auc = average_precision_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, preds)
    print(f"✅ 일반 ROC-AUC: {roc_auc:.4f}")
    print(f"🔥 사기탐지용 극강 지표 PR-AUC (Average Precision): {pr_auc:.4f}")

    # 시각화: 금융 사기 탐지의 핵심인 PR Curve
    precision, recall, _ = precision_recall_curve(y_true, preds)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkred', lw=2, label=f'GAT PR-AUC: {pr_auc:.4f}')
    plt.axhline(y=0.05, color='gray', linestyle='--', label='무작위 추출 (5% 확률)')
    plt.title("금융 불균형 데이터 방어: Precision-Recall Curve")
    plt.xlabel("Recall (실제 사기꾼 중 잡아낸 비율)")
    plt.ylabel("Precision (사기꾼이라고 지목한 사람 중 실제 사기꾼 비율)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("aml_pr_auc_evaluation.png", dpi=150)
    print("✅ 최종 평가 대시보드 저장 완료: aml_pr_auc_evaluation.png")

if __name__ == "__main__":
    run_advanced_pipeline()
