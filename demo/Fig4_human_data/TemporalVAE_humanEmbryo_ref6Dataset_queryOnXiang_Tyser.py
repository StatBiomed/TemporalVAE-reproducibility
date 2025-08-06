# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：TemporalVAE_humanEmbryo_ref6dataset_queryOnTyser.py
@Author  ：awa121
@Date    ：2025/3/21 8:05

Description:
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"project_root: {project_root}")
sys.path.append(project_root)
os.chdir(project_root)

import torch

torch.set_float32_matmul_precision('high')
import pyro

from TemporalVAE.utils import *

# smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
# assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from collections import Counter
import os
import yaml
import argparse
import anndata as ad
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="TemporalVAE")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="/test/Fig4_TemporalVAE_human_ref6dataset_queryOnTyserAndXiang/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/human_embryo_preimplantation/integration_8dataset/",
                        help="sc file folder path.")

    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="50",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")
    # ------------------ model training setting ------------------
    parser.add_argument('--train_epoch_num', type=int,
                        default="50",  # original 100
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=100000,  # original 100000
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg1to1",
                        # default="embryoneg5to5", #  2025-03-27 15:27:56 original
                        help="y_time_nor_train standard type may cause different latent space: "
                             "log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    # supervise_vae            supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")
    # ------------------ task setting ------------------
    # parser.add_argument('--kfold_test', action="store_true", help="(Optional) make the task k fold test on dataset.", default=True)
    # parser.add_argument('--train_whole_model', action="store_true", help="(Optional) use all data to train a model.", default=True)  # necessary here
    # parser.add_argument('--identify_time_cor_gene', action="store_true", help="(Optional) identify time-cor gene by model trained by all.", default=False)

    args = parser.parse_args()

    data_golbal_path = "data/"  # dataset all save at folder "data"
    result_save_path = "results/" + args.result_save_path + "/"
    yaml_path = "vae_model_configs/"
    # ---------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ----------set logger and parameters, creat result save path and folder -----------------------------
    latent_dim = config['model_params']['latent_dim']

    time_standard_type = args.time_standard_type
    sc_data_file_csv = args.file_path + 'rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad'
    _path = '{}/{}/'.format(result_save_path, args.file_path)
    if not os.path.exists(_path):
        os.makedirs(_path)

    logger_file = f'{_path}/{args.vae_param_file}_dim{latent_dim}_time{time_standard_type}_epoch{args.train_epoch_num}_batchSize{args.batch_size}_minGeneNum{args.min_gene_num}.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + args.file_path))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    # query_data_file_csv = "/human_embryo_preimplantation/XiaoCS8/data_count_hvg.csv"
    # query_cell_info_file_csv = "/human_embryo_preimplantation/XiaoCS8/cell_with_time.csv"
    sc_expression_df, cell_time, adata_filtered = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path,
                                                                                sc_data_file_csv,
                                                                                cell_info_file=None,
                                                                                # donor_attr=donor_attr,
                                                                                # drop_out_donor=drop_out_donor,
                                                                                # external_file_name=query_data_file_csv,
                                                                                # external_cell_info_file=query_cell_info_file_csv,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                data_raw_count_bool=True,
                                                                                return_normalized_raw_count=True)  # 2024-04-20 15:38:58

    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["dataset_label"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["day"])))
    save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
    # """2024-08-23 15:57:41 not use here, remove

    #  ---------------------- TASK: use reference data to train a model  ------------------------
    drop_out_donor = ["T", "Xiang"]  #
    print(f"drop the donor: {drop_out_donor}")
    cell_drop_index_list = cell_time.loc[cell_time["dataset_label"].isin(drop_out_donor)].index
    sc_expression_df_filter = sc_expression_df.drop(cell_drop_index_list, axis=0)
    cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
    cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
    sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
        sc_expression_df_filter, donor_dic,
        special_path_str,
        cell_time_filter,
        time_standard_type, config, args.train_epoch_num,
        plot_latentSpaceUmap=False, plot_trainingLossLine=True, time_saved_asFloat=True, batch_dic=batch_dic, donor_str="dataset_label",
        batch_size=int(args.batch_size))  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may have different result
    predict_donors_df = pd.DataFrame(train_clf_result, columns=["pseudotime"], index=cell_time_filter.index)
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                   min(label_dic.values()), max(label_dic.values())))
    cell_time_filter = pd.concat([cell_time_filter, predict_donors_df], axis=1)

    adata_mu_reference = ad.AnnData(X=total_result["mu"].cpu().numpy())
    adata_mu_reference.obs = cell_time_filter[["time", "predicted_time", "dataset_label", "cell_type", "day"]]

    plt_umap_byScanpy(adata_mu_reference.copy(), ["time", "predicted_time", "dataset_label", "cell_type"], save_path=save_path, mode=None, figure_size=(5, 4),
                      color_map="turbo",
                      n_neighbors=50, n_pcs=20, special_file_name_str="n50_")  # color_map="viridis"
    # add reference and query annotation
    adata_mu_reference.obs['data_type'] = 'L & M & P & Z & Xiao & C'
    print(f"{Counter(adata_mu_reference.obs['dataset_label'])}")

    # adata_subset=from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=200, random_seed=0)
    # first use subset of reference data ( 2025-04-14 15:34:22 6 dataset) to train a umap space,
    # second map all reference data, Tyser, Xiang19 to this umap space, respectively.
    # queryOneDataset_referenceOn6Datasets_humanEmbryo("T",
    #                                                  cell_time, sc_expression_df,
    #                                                  time_standard_type, label_dic, batch_dic,
    #                                                  runner, experiment, adata_subset.copy(), _logger, save_path,
    #                                                  special_file_name=f"_subRefTime{random_select_n_timePoint}",umap_n_neighbors=20,
    #                                                  )
    # queryOneDataset_referenceOn6Datasets_humanEmbryo("Xiang",
    #                                                  cell_time, sc_expression_df,
    #                                                  time_standard_type, label_dic, batch_dic,
    #                                                  runner, experiment, adata_subset.copy(), _logger, save_path,
    #                                                  special_file_name=f"_subRefTime{random_select_n_timePoint}",umap_n_neighbors=100,
    #                                                  )
    import umap
    # less cell to train umap reducer.
    umap_n_neighbors = 20
    random_select_n_timePoint = 200
    random_seed = 0
    adata_subset = from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=random_select_n_timePoint, random_seed=random_seed)
    print(f"umap_n_neighbors: {umap_n_neighbors}, random_select_n_timePoint: {random_select_n_timePoint},random seed: {random_seed}")
    umap_reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=0.75, n_components=2, random_state=0)
    adata_subset.obsm['X_umap'] = umap_reducer.fit_transform(adata_subset.X)
    adata_mu_reference.obsm['X_umap'] = umap_reducer.transform(adata_mu_reference.X)
    adata_mu_query_T=queryOneDataset_referenceOn6Datasets_humanEmbryo("T",
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference.copy(), _logger, save_path,
                                                     umap_reducer,
                                                     special_file_name=f"_subCell{random_select_n_timePoint}_umapNei{umap_n_neighbors}"
                                                     )
    adata_mu_query_Xiang=queryOneDataset_referenceOn6Datasets_humanEmbryo("Xiang",
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference.copy(), _logger, save_path,
                                                     umap_reducer,
                                                     special_file_name=f"_subCell{random_select_n_timePoint}_umapNei{umap_n_neighbors}"
                                                     )
    # ----
    _adata_temp = anndata.concat([adata_mu_reference.copy(), adata_mu_query_T.copy(),adata_mu_query_Xiang], axis=0)
    adata_filtered.obsm["X_umap"] = _adata_temp[adata_filtered.obs.index].obsm["X_umap"]
    adata_filtered.obsm['tvae_emb'] = _adata_temp[adata_filtered.obs.index].X
    adata_filtered.obs['tvae_predicted_time'] = _adata_temp[adata_filtered.obs.index].obs["predicted_time"]  # for reconstruction or prediction (optional)
    adata_filtered.write_h5ad(f"{save_path}/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.filtered.h5ad")
    print(f"Save raw count, processed data, umap, low-dim embedding and prediction at {save_path}/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.filtered.h5ad")

    # ---- all cell to train umap reducer
    umap_n_neighbors = 50
    umap_reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=0.75, n_components=2, random_state=0)
    print(f"{Counter(adata_mu_reference.obs['dataset_label'])}")
    print(f"{Counter(adata_mu_reference.obs['time'])}")

    adata_mu_reference.obsm['X_umap'] = umap_reducer.fit_transform(adata_mu_reference.X)
    adata_mu_reference.obsm['X_umap'] = umap_reducer.transform(adata_mu_reference.X)
    queryOneDataset_referenceOn6Datasets_humanEmbryo("Xiang",
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference.copy(), _logger, save_path,
                                                     umap_reducer,
                                                     special_file_name=f"_umapNei{umap_n_neighbors}"
                                                     )
    queryOneDataset_referenceOn6Datasets_humanEmbryo("T",
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference.copy(), _logger, save_path,
                                                     umap_reducer,
                                                     special_file_name=f"_umapNei{umap_n_neighbors}"
                                                     )
    # ----



    # """
    ### ------------TASK : K-FOLD TEST--------------------------------------
    # if args.kfold_test:
    #     queryOneDataset_dropOutOneDataset(["T"],
    #                                       ["Xiang"],
    #                                       cell_time, sc_expression_df, donor_dic, batch_dic, special_path_str, time_standard_type,
    #                                       config, args, _logger, save_file_name
    #                                       )
    #     queryOneDataset_dropOutOneDataset(["Xiang"],
    #                                       ["T"],
    #                                       cell_time, sc_expression_df, donor_dic, batch_dic, special_path_str, time_standard_type,
    #                                       config, args, _logger, save_file_name
    #                                       )

    _logger.info("Finish all.")
    return







# def queryOneDataset_dropOutOneDataset(test_donor_list, drop_out_donor,
#                                       cell_time, sc_expression_df, donor_dic, batch_dic, special_path_str, time_standard_type,
#                                       config, args, _logger, save_file_name,
#                                       k_fold_attr_str="dataset_label"
#                                       ):
#     test_donor = test_donor_list[0]
#     print(f"query on {test_donor_list}, drop {drop_out_donor}")
#
#     cell_drop_index_list = cell_time.loc[cell_time["dataset_label"].isin(drop_out_donor)].index
#     sc_expression_df_filter = sc_expression_df.drop(cell_drop_index_list, axis=0)
#     cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
#     cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
#     predict_donors_dic, label_dic, kFold_result_recall_dic = task_kFoldTest(test_donor_list, sc_expression_df_filter, donor_dic, batch_dic,
#                                                                             special_path_str, cell_time_filter, time_standard_type,
#                                                                             config, args.train_epoch_num, _logger,
#                                                                             donor_str=k_fold_attr_str,
#                                                                             batch_size=args.batch_size, recall_predicted_mu=True)
#     mu_result = kFold_result_recall_dic[test_donor][-1]
#     train_mu_result, test_mu_result = mu_result
#     cell_time_tyser = cell_time.loc[predict_donors_dic[test_donor].index]
#     cell_time_tyser["predicted_time"] = predict_donors_dic[test_donor]['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
#                                                                                                               min(label_dic.values()), max(label_dic.values())))
#     cell_time_tyser = cell_time_tyser[["time", "predicted_time", "dataset_label", "day", "cell_type"]]
#     adata_mu_tyser = ad.AnnData(X=test_mu_result.cpu().numpy(), obs=cell_time_tyser)
#     adata_mu_tyser.obs['data_type'] = test_donor
#
#     # cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
#     # cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
#     # cell_time_referenceDataset=
#     # adata_mu_4dataset=ad.AnnData(X=train_mu_result.cpu().numpy(),obs=cell_time)
#     try:
#         adata_mu_4dataset = ad.read_h5ad(f"{save_file_name}/n50_latent_mu.h5ad")
#     except:
#         print("error on predict on query dataset. \n"
#               "Note: *TASK: use reference data to train a model* is necessary, "
#               "because it generate train's n50_latent_mu.h5ad file. \n"
#               "TASK : K-FOLD TEST is based on Function task_kFoldTest, "
#               "and it's difficult to return lantent_mu of each fold.")
#
#     adata_mu_4dataset.obs['data_type'] = 'L & M & P & Z & Xiao & C'
#     print(f"{Counter(adata_mu_4dataset.obs['dataset_label'])}")
#     # adata_mu_4dataset = downsampling_specificDataset(adata_mu_4dataset.copy(), "C")
#     # adata_mu_4dataset = downsampling_specificDataset(adata_mu_4dataset.copy(), "X")
#
#     adata_all = anndata.concat([adata_mu_4dataset.copy(), adata_mu_tyser.copy()], axis=0)
#
#     adata_all.obs["cell_typeMask4dataset"] = adata_all.obs.apply(lambda row: 'L & M & P & Z & Xiao & C' if row['dataset_label'] != test_donor else row['cell_type'], axis=1)
#     adata_all.obs["cell_typeMaskTyser"] = adata_all.obs.apply(lambda row: test_donor if row['dataset_label'] == test_donor else row['cell_type'], axis=1)
#
#     # ---- 1 method: mapping tyser data to other 4 dataset's umap, just use different umap model
#     # sc.pp.neighbors(adata_mu_tyser, n_neighbors=50, n_pcs=20)
#     # sc.tl.umap(adata_mu_tyser, min_dist=0.75)
#     # ----
#
#     # ----2 method mapping tyser data to other 4 dataset's umap, use same umap model by 4 dataset,
#     # Create a UMAP model instance
#     import umap
#     # reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=101)
#     # reducer = umap.UMAP(n_neighbors=15, min_dist=0.75, n_components=2, random_state=10)
#     reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=0)
#     print(f"{Counter(adata_mu_4dataset.obs['dataset_label'])}")
#
#     embedding_4dataset = reducer.fit_transform(adata_mu_4dataset.X)
#     embedding_tyser = reducer.transform(adata_mu_tyser.X)
#     adata_mu_4dataset.obsm['X_umap'] = embedding_4dataset
#     adata_mu_tyser.obsm['X_umap'] = embedding_tyser
#
#     ### ---------------- Plot images ---------------
#     reference_dataset_str = '&'.join(adata_all.obs['dataset_label'].unique().astype('str'))
#     # combin two AnnData's UMAP loc
#     adata_all.obsm["X_umap"] = np.vstack([adata_mu_4dataset.obsm['X_umap'], adata_mu_tyser.obsm['X_umap']])
#     adata_all.write_h5ad(f"{save_file_name}/{reference_dataset_str}_mu.h5ad")
#     # --- plot on Predict Time
#     plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, label_dic,
#                                                  mask_dataset_label=test_donor, plot_attr='predicted_time',
#                                                  reference_dataset_str=reference_dataset_str,
#                                                  special_file_str=f"_mask{test_donor}_query{test_donor}"
#                                                  )
#     plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, label_dic,
#                                                  mask_dataset_label='L & M & P & Z & Xiao & C',
#                                                  plot_attr='predicted_time',
#                                                  reference_dataset_str=reference_dataset_str,
#                                                  special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}")
#     # --- plot on cell type
#     plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all.copy(), save_file_name, attr="cell_typeMask4dataset",
#                                                           masked_str='L & M & P & Z & Xiao & C', color_palette="tab20",
#                                                           legend_title="Cell type",
#                                                           reference_dataset_str=reference_dataset_str,
#                                                           special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}')
#     plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all.copy(), save_file_name, attr="cell_typeMaskTyser",
#                                                           masked_str=test_donor, color_palette="hsv",
#                                                           legend_title="Cell type",
#                                                           reference_dataset_str=reference_dataset_str,
#                                                           special_file_str=f'_mask{test_donor}_query{test_donor}')
#     # --- plot on dataset
#     plot_tyser_mapping_to_datasets_attrDataset(adata_all.copy(), save_file_name,
#                                                attr="dataset_label", masked_str='T',
#                                                color_dic={'L': "#B292CA",
#                                                           'M': '#7ED957',
#                                                           'P': '#FFC947',
#                                                           'Z': '#00CED1',
#                                                           'Xiao': '#E06377',
#                                                           'C': '#c76f00',
#                                                           test_donor: (0.9, 0.9, 0.9, 0.7)},
#                                                legend_title="Dataset",
#                                                reference_dataset_str=reference_dataset_str,
#                                                special_file_str=f"_mask{test_donor}_query{test_donor}")
#     plot_tyser_mapping_to_datasets_attrDataset(adata_all.copy(), save_file_name,
#                                                attr="data_type", masked_str='L & M & P & Z & Xiao & C',
#                                                color_dic={'L & M & P & Z & Xiao & C': (0.9, 0.9, 0.9, 0.7),
#                                                           test_donor: "#E06D83"},
#                                                reference_dataset_str=reference_dataset_str,
#                                                legend_title="Dataset", special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}")
#
#     # --- plot on time categorical
#     plot_tyser_mapping_to_datasets_attrTimeGT(adata_all.copy(), save_file_name, plot_attr='time',
#                                               query_timePoint='17.5',
#                                               legend_title="Cell stage",
#                                               mask_dataset_label=test_donor,
#                                               reference_dataset_str=reference_dataset_str,
#                                               special_file_str=f'_mask{test_donor}_query{test_donor}')
#     plot_tyser_mapping_to_datasets_attrTimeGT(adata_all.copy(), save_file_name, plot_attr='time',
#                                               query_timePoint='17.5',
#                                               legend_title="Cell stage",
#                                               mask_dataset_label="Liu & Lv & M & P & Z & Xiao & C",
#                                               reference_dataset_str=reference_dataset_str,
#                                               special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}')
#
#     import gc
#     gc.collect()
#     _logger.info("Finish fold-test.")
#
#
# def downsampling_specificDataset(adata, dataset_label):
#     c_mask = adata.obs["dataset_label"] == dataset_label
#     c_indices = np.where(c_mask)[0]
#     total_c = len(c_indices)
#     if total_c > 1000:
#         # 随机选择要删除的索引（保留1000个）
#         keep_indices = np.random.choice(c_indices, size=1000, replace=False)
#         delete_mask = c_mask.copy()
#         delete_mask[keep_indices] = False  # 将要保留的设为False
#
#         # 3. 创建反向选择器（保留所有非"C"或选中的1000个"C"）
#         adata = adata[~delete_mask].copy()
#         print(f"{Counter(adata.obs['dataset_label'])}")
#     else:
#         print(f"数据集'C'只有{total_c}个样本，不足1000个，不做删除")
#     return adata


if __name__ == '__main__':
    main()
