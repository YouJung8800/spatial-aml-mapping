import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Mac 환경 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

def build_and_visualize_aml_network():
    G = nx.DiGraph()
    
    # 1. 일반적인 정상 거래 패턴 (선형적 흐름)
    normal_transactions = [(1, 2), (2, 3), (3, 4), (5, 2), (6, 3), (4, 7)]
    G.add_edges_from(normal_transactions, type='normal')
    
    # 2. 자금세탁 의심 순환(Ring) 거래 패턴 (돈이 돌고 도는 구조)
    suspicious_transactions = [(10, 11), (11, 12), (12, 13), (13, 10)]
    G.add_edges_from(suspicious_transactions, type='suspicious')
    
    # 3. 정상과 의심 그룹 간의 연결 고리 (자금 세탁의 입구)
    G.add_edge(7, 10, type='bridge')

    # 알고리즘: PageRank를 통한 자금 쏠림 현상(위험도) 계산
    pr = nx.pagerank(G, alpha=0.85)
    
    # 시각화 설정
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5) # 노드 간격 조정
    
    # 위험도(PageRank)에 따른 노드 색상 매핑
    node_colors = [pr[node] for node in G.nodes()]
    
    # 노드 및 엣지 그리기
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, 
                                   node_size=700, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='white', font_weight='bold')
    
    # 컬러바 추가 및 디자인
    plt.title("Spatial AML Mapping: 자금세탁 의심 순환 네트워크 탐지", fontsize=16, fontweight='bold')
    cbar = plt.colorbar(nodes, label="위험도 (PageRank 기반 자금 집중도)")
    cbar.set_label("위험도 (PageRank 기반 자금 집중도)", fontsize=12)
    plt.axis("off")
    
    # 결과 저장
    plt.tight_layout()
    plt.savefig("aml_network_dashboard.png", dpi=150, bbox_inches='tight')
    print("✅ AML 네트워크 시각화 대시보드 저장 완료: aml_network_dashboard.png")

if __name__ == "__main__":
    build_and_visualize_aml_network()
