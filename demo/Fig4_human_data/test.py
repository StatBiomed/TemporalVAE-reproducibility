# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：test.py
@Author  ：awa121
@Date    ：2025/9/21 1:09

Description:
"""
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"project_root: {project_root}")
sys.path.append(project_root)
os.chdir(project_root)
import anndata

from TemporalVAE.utils import plot_tyser_mapping_to_datasets_attrDataset

def main(test_donor="T"):
    adata_combined = anndata.read_h5ad(
        f"results/Fig4_TemporalVAE_human_ref6dataset_queryOnTyserAndXiang/human_embryo_preimplantation/integration_8dataset/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg1to1_epoch50_batchSize100000_minGeneNum50/C&Xiao&L&Z&M&P&{test_donor}_mu_subCell200_umapNei20.h5ad")
    save_path='/mnt/yijun/nfs_share/awa_project/awa_github/TemporalVAE/results/test/Fig4_TemporalVAE_human_ref6dataset_queryOnTyserAndXiang_0921_TVAEv1/human_embryo_preimplantation/integration_8dataset/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg1to1_epoch50_batchSize100000_minGeneNum50/'
    # test_donor='Xiang'
    reference_dataset_str=f'C&Xiao&L&Z&M&P&{test_donor}'
    special_file_name='_subCell200_umapNei20'
    # --- plot on dataset
    plot_tyser_mapping_to_datasets_attrDataset(adata_combined.copy(), save_path,
                                               attr="dataset_label", masked_str=test_donor,
                                               color_dic={'L': '#E06377',
                                                          'M': '#7ED957',
                                                          'P': '#FFC947',
                                                          'Z': '#00CED1',
                                                          'Xiao': "#B292CA",
                                                          'C': '#c76f00',
                                                          # 'Lv': '#8f5239',
                                                          test_donor: (0.9, 0.9, 0.9, 0.7)},
                                               legend_title="Dataset",
                                               reference_dataset_str=reference_dataset_str,
                                               special_file_str=f"_mask{test_donor}_query{test_donor}{special_file_name}")
    plot_tyser_mapping_to_datasets_attrDataset(adata_combined.copy(), save_path,
                                               attr="data_type", masked_str='L & M & P & Z & Xiao & C',
                                               color_dic={'L & M & P & Z & Xiao & C': (0.9, 0.9, 0.9, 0.7),
                                                          test_donor: "#E06D83"},
                                               reference_dataset_str=reference_dataset_str,
                                               legend_title="Dataset", special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}{special_file_name}")
    return


if __name__ == '__main__':
    main()
