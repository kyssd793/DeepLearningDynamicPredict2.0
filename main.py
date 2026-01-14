import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import data_preprocess as dp
import model_train_predict as mtp
import os
import tensorflow as tf

from data_preprocess import downsample_data

# 1. 强制暴露所有GPU（关键！）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# 2. 开启显存动态增长
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 3. 重新获取GPU列表（此时能看到所有4块）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 打印所有可用GPU（确认能看到2号）
    print(f"📌 服务器可用GPU列表：{[gpu.name for gpu in gpus]}")

    # 4. 指定绑定2号GPU（索引2）
    target_gpu_idx = 2
    if target_gpu_idx < len(gpus):
        # 只让TensorFlow使用2号GPU
        tf.config.set_visible_devices(gpus[target_gpu_idx], 'GPU')
        # 开启2号GPU的显存动态增长
        tf.config.experimental.set_memory_growth(gpus[target_gpu_idx], True)
        print(f"✅ 已绑定2号GPU: {gpus[target_gpu_idx].name} (4090)")
    else:
        print(f"⚠️ 2号GPU不存在，可用GPU数量：{len(gpus)}")
else:
    print("⚠️ 未检测到GPU，使用CPU运行")


# 主调用逻辑（先降采样，再滤波）
if __name__ == "__main__":
    # 1. 读取原始数据
    csv_path = "fix150.csv"
    raw_df = dp.extract_single_column_csv(csv_path)
    dp.plot_voltage_single_column(raw_df, title_suffix="（原始数据）")

    # 2. 核心步骤1：定向降采样到2w条
    target_count = 20000  # 目标2w条
    down_df_2w, down_sr_2w = dp.downsample_to_target_count(raw_df, target_count=target_count)
    print(f"✅ 降采样到2w条完成，采样率：{down_sr_2w} Hz")
    dp.plot_voltage_single_column(down_df_2w, title_suffix="（定向降采样到2w条后）")
    # mtp.save_one_column_to_csv(down_df_2w, filename="down_sample_fix15.csv")


    # # 3. 核心步骤2：用降采样后的数据找10Hz内的真实基频
    # real_base_freq = dp.plot_spectrum_base_freq(down_df_2w, down_sr_2w, top_n=3)

    # 4. 适用于lstm处理的数据
    # balanced_df = dp.enhance_lstm_feature(down_df_2w, down_sr_2w, 3)
    # 动态基频跟踪预处理（滑窗1000，±1Hz）
    # balanced_df_dynamic = dp.enhance_lstm_feature_dynamic_freq(
    #     down_df_2w,
    #     down_sr_2w,
    #     window_size=2000,
    #     freq_band=1
    # )
    balanced_df = dp.adaptive_piezo_preprocessing(down_df_2w, down_sr_2w)
    print(f"✅ d15_200动态预处理完成")
    print(f"✅ 预处理完成（2w条数据+异常值清理+平衡版平滑）")
    dp.plot_voltage_single_column(balanced_df, title_suffix="（2w条数据+异常值清理+平衡版平滑后）")
    # mtp.save_one_column_to_csv(balanced_df, filename="processed_data_d15-20.csv")


    # # 5. ===================== 训练模型 =====================
    # print("\n===== 开始训练模型 =====")
    # # 模型/scaler保存到当前同级目录（无需创建子文件夹）
    # save_path = './'  # 关键：改为当前目录
    # model, scale = mtp.train_lstm_attention_model_2(
    #     preprocessed_df=balanced_df,
    #     seq_length=32,
    #     save_path=save_path,
    # )
    # print(f"✅ 模型训练完成！")
    #
    # # 6. ===================== 滑窗预测 =====================
    # print("\n===== 开始滑窗预测 =====")
    # # 预测数据用预处理后的balanced_df
    # inputData = balanced_df.copy()
    # time_pred, pred_data = mtp.predict_with_real_window_reset(
    #     dataB=inputData,
    #     model=model,
    #     scaler=scale,
    #     seq_length=32,
    #     # save_path='./',
    #     predict_step_per_round=16,
    #     max_predict_num=20000
    # )
    # ## ===================== 绘图 =====================
    # print("\n===== 生成图片和数据 =====")
    # # 修复1：展平pred_data为一维数组（避免维度不匹配）
    # pred_data_flat = pred_data.flatten()
    #
    # # 新增：填充前32个0，使长度与原始2w条一致
    # seq_length = 32  # 滑动窗口大小
    # pred_data_padded = mtp.pad_pred_data(pred_data_flat, seq_length)
    #
    # # 修复2：确保time_true和time_pred维度匹配
    # time_true = np.arange(len(inputData))
    # time_pred = np.arange(len(pred_data_padded))  # 填充后预测数据的时间轴
    #
    # # =====================只会绘图不存 =====================
    # print("\n===== 绘制真实/预测对比图 =====")
    # mtp.plot_double_figure(
    #     true_data=down_df_2w,       # 真实数据（balanced_df）
    #     time_true=time_true,       # 真实数据的采样点序号
    #     pred_data=pred_data_padded,  # 展平后的预测数据
    #     time_pred=time_pred        # 预测数据的采样点序号
    # )
    #
    # ## 存储为.csv文件
    # mtp.save_one_column_to_csv(pred_data_padded, filename='prediction_result_d15-20.csv')
    # ## 存储png图像
    # mtp.save_comparison_plot(
    #     true_data=inputData,
    #     pred_data=pred_data_padded,
    #     time_true=time_true,
    #     time_pred=time_pred,
    #     filename='prediction_comparison_d15-20.png'
    # )
    # print("\n===== 执行完成！生成的文件： =====")
    # print(f"1. 预测数据CSV：{os.path.abspath('prediction_result_d15-20.csv')}")
    # print(f"2. 对比图PNG：{os.path.abspath('prediction_comparison_d15-20.png')}")

    # ===================== 第四步：保存预测结果（同级目录，无绘图） =====================
    # print("\n===== 保存预测结果 =====")
    # # 保存到当前目录的prediction_result.csv
    # mtp.save_to_csv(
    #     time_data=time_pred,  # 采样点序号（替代时间）
    #     predicted_data_inversed=pred_data,
    #     filename='prediction_result.csv'  # 同级目录保存
    # )
    # print(f"✅ 预测结果已保存至：{os.path.abspath('prediction_result.csv')}")
    #
    # # 打印关键信息（方便验证）
    # print(f"\n===== 执行完成关键信息 =====")
    # print(f"模型文件：{os.path.abspath('lstm_attention_piezo_model')}")
    # print(f"Scaler文件：{os.path.abspath('scaler_piezo.pkl')}")
    # print(f"预测结果文件：{os.path.abspath('prediction_result.csv')}")
    # print(f"预测点数：{len(pred_data)} 个，采样点序号范围：{time_pred.min()} ~ {time_pred.max()}")
