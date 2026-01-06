import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import data_preprocess as dp
import model_train_predict as mtp
import os
import tensorflow as tf

# 提前设置GPU可见性（在所有TensorFlow操作之前执行）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 只使用第一个GPU（和原逻辑一致）
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ 已绑定GPU: {gpus[0]}")
else:
    print("⚠️ 未检测到GPU，使用CPU运行")


# 主调用逻辑
if __name__ == "__main__":
    # 1. 读取原始数据
    csv_path = "fix150.csv"
    raw_df = dp.extract_single_column_csv(csv_path)
    # dp.plot_voltage_single_column(raw_df, title_suffix="（原始数据）")


    # 2. 核心步骤1：定向降采样到2w条（优先执行！）
    target_count = 20000  # 目标2w条
    down_df_2w, down_sr_2w = dp.downsample_to_target_count(raw_df, target_count=target_count)
    print(f"✅ 降采样到2w条完成，采样率：{down_sr_2w} Hz")
    # dp.plot_voltage_single_column(down_df_2w, title_suffix="（定向降采样到2w条后）")

    # 3. 核心步骤2：用降采样后的数据找10Hz内的真实基频
    real_base_freq = dp.plot_spectrum_base_freq(down_df_2w, down_sr_2w, top_n=3)


    # # 2. 核心步骤：用原始数据找10Hz内的真实基频（关键！）
    # raw_sr = 50000
    # real_base_freq = dp.plot_spectrum_base_freq(raw_df, raw_sr, top_n=3)

    # 4. 平衡版平滑（保留尖峰+渐变）
    balanced_df = dp.wavelet_denoise_with_envelope_fix(down_df_2w, wavelet='db4', level=3)
    print(f"✅ 预处理完成（2w条数据+异常值清理+平衡版平滑）")
    # dp.plot_voltage_single_column(balanced_df, title_suffix="（2w条数据+异常值清理+平衡版平滑后）")

    # # 3. 峰值降采样（保留脉冲尖峰）
    # down_df, down_sr = dp.downsample_data_peak_preserve_signed(raw_df, downsample_factor=5)
    # dp.plot_voltage_single_column(down_df, title_suffix="（峰值降采样后，未降噪）")
    #
    # # # 4. 新增：异常值清理（删除突然的异常尖峰）
    # # cleaned_df = remove_sudden_spikes(down_df, slope_threshold=10)
    # # plot_voltage_single_column(cleaned_df, title_suffix="（异常值清理后）")
    #
    # # 5. 平衡版平滑（保留尖峰+渐变）
    # balanced_df = dp.wavelet_denoise_with_envelope_fix(down_df, wavelet='db4', level=3)
    # print(f"✅ 预处理完成（异常值清理+平衡版平滑）")
    # dp.plot_voltage_single_column(balanced_df, title_suffix="（异常值清理+平衡版平滑后）")

    # ===================== 2. 训练模型 =====================
    print("\n===== 开始训练模型 =====")
    # 模型/scaler保存到当前同级目录（无需创建子文件夹）
    save_path = './'  # 关键：改为当前目录

    # 调用滚动训练函数
    model, scaler = mtp.train_lstm_attention_model(
        preprocessed_df=balanced_df,  # 预处理后的Voltage列数据
        seq_length=32,  # 时间步长（和你一致）
        save_path=save_path,  # 保存到当前目录
        roll_window_ratio=0.2,  # 验证窗口20%
        roll_step_ratio=0.1  # 滚动步长10%
    )
    print(f"✅ 模型训练完成！模型文件：{os.path.join(save_path, 'lstm_attention_piezo_model')}")

    # ===================== 第三步：滑窗预测（单独执行，用同级目录的模型） =====================
    print("\n===== 开始滑窗预测 =====")
    # 预测数据用预处理后的balanced_df（也可替换为新测试数据）
    dataB = balanced_df.copy()

    # # 调用适配后的预测函数（无Time(s)列，采样点序号为时间轴）
    # time_pred, pred_data = mtp.predict_with_sliding_window_fixed(
    #     dataB=dataB,
    #     seq_length=32,
    #     model_path='./lstm_attention_piezo_model',  # 同级目录的模型
    #     scaler_path='./scaler_piezo.pkl',  # 同级目录的scaler
    #     future_steps=16,  # 单次预测16点（你的核心逻辑）
    # )
    # 在main.py的预测阶段替换为：
    time_pred, pred_data = mtp.predict_stepped_window_fast(
        dataB=dataB,
        seq_length=32,  # 输入窗口32个点
        model_weights_path='./lstm_model_weights.h5',
        scaler_path='./scaler_piezo.pkl',
        predict_step=32,  # 单次预测32个点（和窗口等长）
        target_total_points=20000  # 目标预测20万点（和原始数据一致）
    )
    # ===================== 绘图 =====================
    print("\n===== 生成图片和数据 =====")
    # 修复1：展平pred_data为一维数组（避免维度不匹配）
    pred_data_flat = pred_data.flatten()
    # 修复2：确保time_true和time_pred维度匹配
    time_true = np.arange(len(dataB))

    mtp.save_prediction_to_csv(pred_data_flat, filename='prediction_result_fix15.csv')
    mtp.save_comparison_plot(
        true_data=dataB,
        pred_data=pred_data_flat,
        time_true=time_true,
        time_pred=time_pred,
        filename='prediction_comparison.png'
    )
    print("\n===== 执行完成！生成的文件： =====")
    print(f"1. 预测数据CSV：{os.path.abspath('prediction_result.csv')}")
    print(f"2. 对比图PNG：{os.path.abspath('prediction_comparison.png')}")

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
