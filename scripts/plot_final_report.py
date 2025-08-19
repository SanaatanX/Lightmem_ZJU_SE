import matplotlib.pyplot as plt
import numpy as np



# --- 实验A (r=8) ---
# WikiUpdate 数据
wiki_trr = np.array([0, 35.17, 40.29, 64.47, 71.78, 89.68, -84.48])
wiki_scores = np.array([3.2272, 3.2367, 3.2087, 2.8771, 2.6777, 2.2287, 2.7512])
wiki_labels = ['original_text', 'summ_l1_slight', 'summ_l2_light', 'summ_l3_medium', 'summ_l4_heavy', 'summ_l5_extreme',
               'ext']

# LongMemEval 数据
longmem_trr = np.array([0, 93.82, 96.12, 98.04, 98.58, 99.46, 82.05, ])
longmem_scores_percent = np.array([5.4, 4.4, 21.2, 16.4, 14.8, 11.4, 11.6])
longmem_labels = ['summ_l1_slight', 'summ_l2_light', 'summ_l3_medium', 'summ_l4_heavy', 'summ_l5_extreme', 'ext']

# --- 实验B (r=8, 完整数据点) ---
full_capacity_items = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
full_r8_capacity_scores = np.array([3.1951, 3.1945, 3.1839, 3.2238, 3.2625, 3.2074, 3.2415, 3.2174, 3.2259, 3.2224])

# --- 正交实验 (任务2) ---
# 2a - Rank vs. Quantity
lite_capacity_items = np.array([100, 300, 500])
r4_capacity_scores = np.array([3.1950, 3.2089, 3.2039])
r16_capacity_scores = np.array([3.2145, 3.2247, 3.2589])

# 2b - Rank vs. Compression
comp_labels = ['Original Text\n(TRR=0%)', 'Medium Summary\n(TRR=64.5%)', 'Extreme Summary\n(TRR=89.7%)']
r4_comp_scores = np.array([3.2494, 2.9120, 2.2474])
r8_comp_scores = np.array([3.2272, 2.8771, 2.2287])  # 使用实验A中r=8的对应数据
r16_comp_scores = np.array([3.2368, 2.8803, 2.2120])

# --- 定义基准线的值 ---
WIKI_BASELINE_SCORE = 3.1700
LONGMEM_BASELINE_ACCURACY = 5.0  # 单位是 %

# ==============================================================================
# --- 开始绘图 ---
# ==============================================================================

plt.style.use('seaborn-v0_8-whitegrid')

# --- 图表1：实验A - WikiUpdate 性能衰减曲线 ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(wiki_trr[:-1], wiki_scores[:-1], 'o-', label='WikiUpdate (Summarization)', linewidth=2.5, markersize=8)
ax1.scatter(wiki_trr[-1], wiki_scores[-1], c='red', s=120, label='WikiUpdate (Extraction)', zorder=5, marker='X')
ax1.axhline(y=WIKI_BASELINE_SCORE, color='gray', linestyle='--', linewidth=2,
            label=f'Baseline Score ({WIKI_BASELINE_SCORE})')
ax1.set_title('Exp A-1: Performance vs. TRR on WikiUpdate (r=8)', fontsize=16, weight='bold')
ax1.set_xlabel('Token Reduction Rate (TRR %)', fontsize=14)
ax1.set_ylabel('Average Model Score (out of 5)', fontsize=14)
ax1.annotate('Inflection Point', xy=(64.47, 2.8771), xytext=(55, 3.0),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
ax1.legend(fontsize=12)
plt.tight_layout()
plt.savefig("fig1_exp_a_wiki.png", dpi=300)
print("Chart 1 (fig1_exp_a_wiki.png) with baseline saved.")

# --- 图表2：实验A - LongMemEval 性能衰减曲线 ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.plot(longmem_trr[:-1], longmem_scores_percent[:-1], 'o-', label='LongMemEval (Summarization)', color='purple',
         linewidth=2.5, markersize=8)
ax2.scatter(longmem_trr[-1], longmem_scores_percent[-1], c='magenta', s=120, label='LongMemEval (Extraction)', zorder=5,
            marker='X')
ax2.axhline(y=LONGMEM_BASELINE_ACCURACY, color='gray', linestyle='--', linewidth=2,
            label=f'Baseline Accuracy ({LONGMEM_BASELINE_ACCURACY}%)')
ax2.set_title('Exp A-2: Performance vs. TRR on LongMemEval (r=8)', fontsize=16, weight='bold')
ax2.set_xlabel('Token Reduction Rate (TRR %)', fontsize=14)
ax2.set_ylabel('Accuracy (%)', fontsize=14)
ax2.annotate('Performance Peak\n(Aha Moment)', xy=(96.12, 21.2), xytext=(90, 18),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
ax2.legend(fontsize=12)
plt.tight_layout()
plt.savefig("fig2_exp_a_longmem.png", dpi=300)
print("Chart 2 (fig2_exp_a_longmem.png) with baseline saved.")

# --- 图表3：实验2a - Rank vs. 知识数量 (高分辨率版) ---
fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.plot(lite_capacity_items, r4_capacity_scores, 'o--', label='Rank = 4', alpha=0.7, markersize=8)
ax3.plot(full_capacity_items, full_r8_capacity_scores, 's-', label='Rank = 8 (Full Data)', linewidth=2.0,
         markersize=8)  # 使用完整数据
ax3.plot(lite_capacity_items, r16_capacity_scores, '^-', label='Rank = 16', linewidth=2.5, markersize=8)
ax3.axhline(y=WIKI_BASELINE_SCORE, color='red', linestyle='--', linewidth=2,
            label=f'Baseline Score ({WIKI_BASELINE_SCORE})')
ax3.set_title('Exp 2a: Rank vs. Knowledge Quantity', fontsize=16, weight='bold')
ax3.set_xlabel('Number of Training Items', fontsize=14)
ax3.set_ylabel('Average Model Score (out of 5)', fontsize=14)
ax3.set_xticks(full_capacity_items)
ax3.tick_params(axis='x', rotation=30)
ax3.legend(fontsize=12)
plt.tight_layout()
plt.savefig("fig3_exp_2a_capacity.png", dpi=300)
print("Chart 3 (fig3_exp_2a_capacity.png) with high-resolution r=8 curve and baseline saved.")

# --- 图表4：实验2b - Rank vs. 压缩级别 ---
fig4, ax4 = plt.subplots(figsize=(12, 7))
x = np.arange(len(comp_labels))
width = 0.25
rects1 = ax4.bar(x - width, r4_comp_scores, width, label='Rank = 4', color='skyblue')
rects2 = ax4.bar(x, r8_comp_scores, width, label='Rank = 8', color='royalblue')
rects3 = ax4.bar(x + width, r16_comp_scores, width, label='Rank = 16', color='navy')
ax4.axhline(y=WIKI_BASELINE_SCORE, color='red', linestyle='--', linewidth=2,
            label=f'Baseline Score ({WIKI_BASELINE_SCORE})')
ax4.set_title('Exp 2b: Rank vs. Compression Level', fontsize=16, weight='bold')
ax4.set_ylabel('Average Model Score (out of 5)', fontsize=14)
ax4.set_xticks(x)
ax4.set_xticklabels(comp_labels, fontsize=12, rotation=0)
ax4.legend(fontsize=12)
plt.tight_layout()
plt.savefig("fig4_exp_2b_compression.png", dpi=300)
print("Chart 4 (fig4_exp_2b_compression.png) with baseline saved.")

plt.close('all')  # 关闭所有图像窗口

print("\nAll 4 final report charts have been successfully generated.")
