import data_preprocess as dp
import model_train_predict as mtp

# 主调用逻辑
if __name__ == "__main__":
    # 1. 读取原始数据
    csv_path = "fix150.csv"
    raw_df = dp.extract_single_column_csv(csv_path)
    dp.plot_voltage_single_column(raw_df, title_suffix="（原始数据）")

    # 2. 核心步骤：用原始数据找10Hz内的真实基频（关键！）
    raw_sr = 50000
    real_base_freq = dp.plot_spectrum_base_freq(raw_df, raw_sr, top_n=3)

    # 3. 峰值降采样（保留脉冲尖峰）
    down_df, down_sr = dp.downsample_data_peak_preserve_signed(raw_df, downsample_factor=5)
    dp.plot_voltage_single_column(down_df, title_suffix="（峰值降采样后，未降噪）")

    # # 4. 新增：异常值清理（删除突然的异常尖峰）
    # cleaned_df = remove_sudden_spikes(down_df, slope_threshold=10)
    # plot_voltage_single_column(cleaned_df, title_suffix="（异常值清理后）")

    # 5. 平衡版平滑（保留尖峰+渐变）
    balanced_df = dp.wavelet_denoise_with_envelope_fix(down_df, wavelet='db4', level=3)
    print(f"✅ 预处理完成（异常值清理+平衡版平滑）")
    dp.plot_voltage_single_column(balanced_df, title_suffix="（异常值清理+平衡版平滑后）")

    # # 5. 可选：小波变换补充验证（贴合导师建议，提升学术性）
    # wavelet_verify_base_freq_simple(raw_df, real_base_freq, raw_sr)