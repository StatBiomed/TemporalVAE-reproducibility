# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：plot_figureC.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024/9/6 22:04
"""
# -*-coding:utf-8 -*-
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"project_root: {project_root}")
sys.path.append(project_root)
os.chdir(project_root)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    data = {
        'Method': ['TemporalVAE', 'TemporalVAE', 'TemporalVAE',
                   'Calderon22', 'Calderon22', 'Calderon22',
                   'LR', 'LR', 'LR',
                   'PCA', 'PCA', 'PCA',
                   "RF", "RF", "RF"],
        'Correlation Type': ['Spearman', 'Pearson', 'Kendall’s τ',
                             'Spearman', 'Pearson', 'Kendall’s τ',
                             'Spearman', 'Pearson', 'Kendall’s τ',
                             'Spearman', 'Pearson', 'Kendall’s τ',
                             'Spearman', 'Pearson', 'Kendall’s τ', ],
        'Value': [0.89012, 0.91359, 0.72915,
                  0.83787, 0.79336, 0.65749,
                  0.78595, 0.79398, 0.60049,
                  0.25161, 0.14779, 0.16165,
                  0.07168, 0.36478, 0.15780]
    }


    # for "embryo_1"
    # time_dic={"TemporalVAE":2161.1639816761017,}
    df = pd.DataFrame(data)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建条形图
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x='Method', y='Value', hue='Correlation Type', data=df, palette=["#D1B2FF", "#FFCBA4", "#A6E3D7"])

    # 添加标题和坐标轴标签
    plt.title('', fontsize=16)
    plt.ylabel('Correlation value', fontsize=16)
    plt.xlabel('', fontsize=15)

    # 调整图例
    plt.legend(title='Correlation Type', title_fontsize='16', fontsize='16')

    # 在每个条形上显示数值
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 10),
                         textcoords='offset points')

    # 设置Y轴范围以清晰展示负相关性，添加Y=0参考线强调正负相关性
    plt.ylim(0, 1)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    # 设置x轴标签的字体大小
    plt.tick_params(axis='x', labelsize=16)  # Set x-axis label size to 14

    # 美化图表
    # sns.despine(offset=10, trim=True)  # 减少边框
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.savefig(f"results/Fig3_TemporalVAE_kfoldOn_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100/compareWithBaselineMethods.png", dpi=450, ) # before 2025-09-17 22:41:41 dpi is 300
    plt.savefig(f"results/Fig3_TemporalVAE_kfoldOn_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100/compareWithBaselineMethods.pdf", dpi=450, ) # before 2025-09-17 22:41:41 dpi is 300
    plt.show()
    plt.close()
    print(f"figure save as results/Fig3_TemporalVAE_kfoldOn_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100/compareWithBaselineMethods.png")
    plt.show()




if __name__ == '__main__':
    main()
