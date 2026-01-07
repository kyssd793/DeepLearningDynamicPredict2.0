import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Attention
from tensorflow.keras.models import Model
import tensorflow as tf
import pickle


def save_to_csv(time_data, predicted_data_inversed, filename):
    # 如果是二维数组或其他类型，可以展平为一维数组
    time_data = time_data.flatten() if hasattr(time_data, 'flatten') else time_data
    predicted_data_inversed = predicted_data_inversed.flatten() if hasattr(predicted_data_inversed, 'flatten') else predicted_data_inversed

    # 创建 DataFrame
    df = pd.DataFrame({
        'time': time_data,
        'predicted': predicted_data_inversed
    })

    # 保存为 CSV 文件
    df.to_csv(filename, index=False)


def save_comparison_plot(true_data, pred_data, time_true, time_pred,
                            filename='prediction_comparison.png', dpi=300):
    # 创建画布（适配大量数据的尺寸）
    fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)

    # 绘制真实数据（绿色）
    ax.plot(time_true, true_data['Voltage'].values,
            color='#2ecc71', label='True Data', alpha=0.8, linewidth=0.8)
    # 绘制预测数据（红色）
    ax.plot(time_pred, pred_data,
            color='#e74c3c', label='Predicted Data', alpha=0.8, linewidth=0.8)

    # 图表配置（英文标签，避免字体问题）
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title('True vs Predicted Voltage Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 紧凑布局，避免裁剪
    plt.tight_layout()
    # 保存图片（高清，无白边）
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭画布，释放内存

    print(f"✅ 对比图已保存为PNG：{os.path.abspath(filename)}")
    return os.path.abspath(filename)

def save_prediction_to_csv(pred_data, filename='prediction_result.csv'):
    # 展平数据为一维，确保单列
    pred_data_flat = pred_data.flatten()
    # 创建单列DataFrame
    df = pd.DataFrame({
        'Predicted_Voltage': pred_data_flat
    })
    # 保存CSV（无索引，仅数据）
    df.to_csv(filename, index=False, header=False)
    print(f"✅ 预测数据已保存为CSV：{os.path.abspath(filename)}")
    return os.path.abspath(filename)

def ensemble_loss(y_true, y_pred, a=0.3, b=0.7):
    # 自定义集成损失函数
    loss1 = tf.square(y_true - y_pred)
    loss2 = tf.abs(y_true - y_pred)
    return a * loss1 + b * loss2


# 全局标准化函数（训练和预测用同一个scaler）
def create_scaler(data):
    """基于训练数据创建scaler，供后续预测使用"""
    scaler = MinMaxScaler()
    scaler.fit(data[['Voltage']])
    return scaler


# 构建数据集（不变）
def create_sequences(data_scaled, seq_length=32):
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length, 0])
        y.append(data_scaled[i + seq_length, 0])

    if len(X) == 0:
        raise ValueError(f"数据长度不足！至少需要 {seq_length + 1} 个样本（当前：{len(data_scaled)}）")

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


# 统一模型构建函数（训练/预测共用，避免结构不一致）
def build_unified_model(seq_length=32):
    """统一的模型结构，训练和预测都用这个"""
    inputs = Input(shape=(seq_length, 1))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv2)
    lstm1 = LSTM(units=50, return_sequences=True)(pool1)
    attention_out = Attention()([lstm1, lstm1])
    lstm2 = LSTM(units=50, return_sequences=False)(attention_out)
    lstm2 = Dropout(0.1)(lstm2)
    outputs = Dense(1)(lstm2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=ensemble_loss)
    return model


# 不太行，已经弃之不用。
def train_lstm_attention_model(preprocessed_df, seq_length=32, save_path='./',
                               roll_window_ratio=0.2, roll_step_ratio=0.1):
    """
    核心改进：优先加载已有模型/Scaler，无文件时才重新训练
    :return: 模型、scaler
    """
    # 模型/Scaler路径
    model_weights_path = os.path.join(save_path, 'lstm_model_weights.h5')
    scaler_path = os.path.join(save_path, 'scaler_piezo.pkl')

    # ========== 关键：优先加载已有文件 ==========
    if os.path.exists(model_weights_path) and os.path.exists(scaler_path):
        # 加载已有Scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        # 加载已有模型
        model = build_unified_model(seq_length)
        model.load_weights(model_weights_path)
        print(f"✅ 已加载已有模型：{model_weights_path}")
        print(f"✅ 已加载已有Scaler：{scaler_path}")
        return model, scaler

    # ========== 无文件时才重新训练 ==========
    print("⚠️ 未检测到已有模型/Scaler，开始重新训练...")
    scaler = create_scaler(preprocessed_df)
    data_scaled = scaler.transform(preprocessed_df[['Voltage']])

    # 构建完整序列数据集
    X_all, y_all = create_sequences(data_scaled, seq_length)
    total_samples = len(X_all)

    # 滚动验证参数计算
    val_window_size = int(total_samples * roll_window_ratio)
    roll_step = int(total_samples * roll_step_ratio)
    if val_window_size == 0 or roll_step == 0:
        raise ValueError("数据量过小，无法进行滚动验证！请增大数据量或调整窗口/步长比例")

    # 初始化模型
    model = build_unified_model(seq_length)

    # 滚动验证训练
    start_idx = 0
    val_loss_list = []
    while start_idx + val_window_size <= total_samples:
        # 划分训练/验证窗口
        train_end_idx = start_idx
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_window_size

        X_train = X_all[:train_end_idx] if train_end_idx > 0 else X_all[:val_start_idx]
        y_train = y_all[:train_end_idx] if train_end_idx > 0 else y_all[:val_start_idx]
        X_val = X_all[val_start_idx:val_end_idx]
        y_val = y_all[val_start_idx:val_end_idx]

        # 跳过样本不足的情况
        if len(X_train) < 100 or len(X_val) < 50:
            start_idx += roll_step
            continue

        # 训练当前窗口
        print(f"\n=== 滚动窗口 {start_idx // roll_step + 1} ===")
        print(f"训练集：0 ~ {train_end_idx if train_end_idx > 0 else val_start_idx} 样本")
        print(f"验证集：{val_start_idx} ~ {val_end_idx} 样本")

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=False
        )

        # 记录验证损失
        val_loss = history.history['val_loss'][-1]
        val_loss_list.append(val_loss)
        print(f"当前窗口验证损失：{val_loss:.4f}")

        # 滚动到下一个窗口
        start_idx += roll_step

    # 保存训练好的模型和scaler
    model.save_weights(model_weights_path)
    print(f"\n✅ 模型权重已保存：{model_weights_path}")
    print(f"平均验证损失：{np.mean(val_loss_list):.4f}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler已保存：{scaler_path}")

    return model, scaler


# 核心：重构后的滑动窗口训练函数
def train_lstm_attention_model_2(preprocessed_df, seq_length=32, save_path='./',
                               window_ratio=0.1, val_ratio=0.2, step_ratio=0.5):
    """
    适配时序特征变化的滑动窗口训练函数
    :param preprocessed_df: 预处理后的DataFrame（含Voltage列）
    :param seq_length: 序列窗口长度
    :param save_path: 模型保存路径
    :param window_ratio: 训练窗口占总样本的比例（默认10%）
    :param val_ratio: 验证集占训练窗口的比例（默认20%）
    :param step_ratio: 窗口滑动步长占训练窗口的比例（默认50%）
    :return: 最优模型、全局scaler
    """
    # 模型/Scaler路径（保留原有命名）
    model_weights_path = os.path.join(save_path, 'lstm_model_weights.h5')
    scaler_path = os.path.join(save_path, 'scaler_piezo.pkl')

    # ========== 优先加载已有模型（保留原有逻辑） ==========
    if os.path.exists(model_weights_path) and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        model = build_unified_model(seq_length)
        model.load_weights(model_weights_path)
        print(f"✅ 已加载已有模型：{model_weights_path}")
        print(f"✅ 已加载已有Scaler：{scaler_path}")
        return model, scaler

    # ========== 滑动窗口训练核心逻辑 ==========
    print("⚠️ 未检测到已有模型/Scaler，开始滑动窗口训练...")

    # 全局标准化（保留原有逻辑，如需局部标准化可在此修改）
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(preprocessed_df[['Voltage']])
    X_all, y_all = create_sequences(data_scaled, seq_length)
    total_samples = len(X_all)

    # 计算滑动窗口参数（适配时序特征变化）
    window_size = int(total_samples * window_ratio)  # 训练窗口大小
    val_size = int(window_size * val_ratio)  # 验证集大小
    step = int(window_size * step_ratio)  # 窗口滑动步长

    # 边界校验
    if window_size < 100 or val_size < 20:
        raise ValueError("窗口过小！请增大window_ratio或确保数据量充足")
    if window_size + val_size >= total_samples:
        raise ValueError("窗口+验证集超过总样本！请减小window_ratio/val_ratio")

    best_val_loss = float('inf')
    best_model_weights = None

    # 滑动窗口训练循环
    for start in range(0, total_samples - window_size - val_size, step):
        # 1. 划分当前窗口的训练/验证集（纯局部数据，非累加）
        X_train = X_all[start:start + window_size]
        y_train = y_all[start:start + window_size]
        X_val = X_all[start + window_size:start + window_size + val_size]
        y_val = y_all[start + window_size:start + window_size + val_size]

        # 2. 每个窗口重新初始化模型（避免增量过拟合）
        model = build_unified_model(seq_length)

        # 3. 少量epoch训练（时序窗口避免过拟合）
        print(f"\n=== 滑动窗口 {start // step + 1} ===")
        print(f"训练窗口：{start} ~ {start + window_size}（共{len(X_train)}样本）")
        print(f"验证窗口：{start + window_size} ~ {start + window_size + val_size}（共{len(X_val)}样本）")

        history = model.fit(
            X_train, y_train,
            epochs=3,  # 时序窗口训练epoch不宜过多
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=False  # 时序数据禁止shuffle
        )

        # 4. 记录最优模型（保留验证损失最低的）
        current_val_loss = history.history['val_loss'][-1]
        print(f"当前窗口验证损失：{current_val_loss:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_weights = model.get_weights()
            print(f"📈 更新最优模型（验证损失：{best_val_loss:.4f}）")

    # ========== 保存最优模型 ==========
    if best_model_weights is None:
        raise RuntimeError("无有效训练窗口！请检查数据量或窗口参数")

    # 加载最优权重并保存
    final_model = build_unified_model(seq_length)
    final_model.set_weights(best_model_weights)
    final_model.save_weights(model_weights_path)

    # 保存scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ 滑动窗口训练完成！")
    print(f"📊 最优验证损失：{best_val_loss:.4f}")
    print(f"💾 模型保存至：{model_weights_path}")
    print(f"💾 Scaler保存至：{scaler_path}")

    return final_model, scaler


def train_lstm_attention_model_local_scaler(preprocessed_df, seq_length=32, save_path='./',
                                            window_ratio=0.1, val_ratio=0.2, step_ratio=0.5):
    """
    适配时序特征变化的滑动窗口训练函数（局部窗口标准化）
    :param preprocessed_df: 预处理后的DataFrame（含Voltage列）
    :param seq_length: 序列窗口长度
    :param save_path: 模型保存路径
    :param window_ratio: 训练窗口占总样本的比例（默认10%）
    :param val_ratio: 验证集占训练窗口的比例（默认20%）
    :param step_ratio: 窗口滑动步长占训练窗口的比例（默认50%）
    :return: 最优模型、各窗口scaler字典、全局参考scaler（用于兜底）
    """
    # 模型/Scaler路径（新增窗口scaler保存）
    model_weights_path = os.path.join(save_path, 'lstm_model_weights_local.h5')
    window_scalers_path = os.path.join(save_path, 'window_scalers.pkl')
    global_scaler_path = os.path.join(save_path, 'global_scaler_piezo.pkl')

    # ========== 优先加载已有模型（兼容局部scaler） ==========
    if os.path.exists(model_weights_path) and os.path.exists(window_scalers_path) and os.path.exists(
            global_scaler_path):
        with open(window_scalers_path, 'rb') as f:
            window_scalers = pickle.load(f)
        with open(global_scaler_path, 'rb') as f:
            global_scaler = pickle.load(f)
        model = build_unified_model(seq_length)
        model.load_weights(model_weights_path)
        print(f"✅ 已加载已有模型：{model_weights_path}")
        print(f"✅ 已加载窗口scaler：{window_scalers_path}")
        print(f"✅ 已加载全局参考scaler：{global_scaler_path}")
        return model, window_scalers, global_scaler

    # ========== 滑动窗口训练核心逻辑（局部标准化） ==========
    print("⚠️ 未检测到已有模型/Scaler，开始局部标准化滑动窗口训练...")

    # 先构建全局序列（用原始数据，不做全局标准化）
    raw_data = preprocessed_df[['Voltage']].values
    total_samples_raw = len(raw_data) - seq_length
    if total_samples_raw < 100:
        raise ValueError("原始数据量不足！至少需要100个有效样本")

    # 计算滑动窗口参数（基于原始数据）
    window_size = int(total_samples_raw * window_ratio)  # 训练窗口大小
    val_size = int(window_size * val_ratio)  # 验证集大小
    step = int(window_size * step_ratio)  # 窗口滑动步长

    # 边界校验
    if window_size < 100 or val_size < 20:
        raise ValueError("窗口过小！请增大window_ratio或确保数据量充足")
    if window_size + val_size >= total_samples_raw:
        raise ValueError("窗口+验证集超过总样本！请减小window_ratio/val_ratio")

    best_val_loss = float('inf')
    best_model_weights = None
    window_scalers = {}  # 保存每个窗口的scaler：{窗口起始位置: scaler对象}

    # 滑动窗口训练循环
    for start in range(0, total_samples_raw - window_size - val_size, step):
        # 1. 取当前窗口的原始数据（非标准化）
        # 原始数据的索引：序列起始位置 = start → 序列结束位置 = start + window_size + val_size + seq_length
        raw_start = start
        raw_end_train = start + window_size + seq_length
        raw_end_val = raw_end_train + val_size

        # 训练窗口原始数据
        raw_train_data = raw_data[raw_start:raw_end_train]
        # 验证窗口原始数据
        raw_val_data = raw_data[raw_end_train - seq_length:raw_end_val]

        # 2. 局部标准化：每个窗口单独训练scaler
        local_scaler = MinMaxScaler()
        train_scaled = local_scaler.fit_transform(raw_train_data)
        val_scaled = local_scaler.transform(raw_val_data)

        # 3. 构建当前窗口的序列
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_val, y_val = create_sequences(val_scaled, seq_length)

        # 跳过样本不足的情况
        if len(X_train) < 50 or len(X_val) < 10:
            print(f"⚠️ 窗口{start // step + 1}样本不足，跳过")
            continue

        # 4. 每个窗口重新初始化模型（避免增量过拟合）
        model = build_unified_model(seq_length)

        # 5. 少量epoch训练（时序窗口避免过拟合）
        print(f"\n=== 滑动窗口 {start // step + 1} ===")
        print(f"训练窗口原始数据：{raw_start} ~ {raw_end_train}（共{len(raw_train_data)}样本）")
        print(f"验证窗口原始数据：{raw_end_train - seq_length} ~ {raw_end_val}（共{len(raw_val_data)}样本）")
        print(f"训练序列数：{len(X_train)} | 验证序列数：{len(X_val)}")

        history = model.fit(
            X_train, y_train,
            epochs=3,  # 时序窗口训练epoch不宜过多
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=False  # 时序数据禁止shuffle
        )

        # 6. 记录当前窗口scaler和最优模型
        window_scalers[start] = local_scaler
        current_val_loss = history.history['val_loss'][-1]
        print(f"当前窗口验证损失：{current_val_loss:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_weights = model.get_weights()
            print(f"📈 更新最优模型（验证损失：{best_val_loss:.4f}）")

    # ========== 保存最优模型和scaler ==========
    if best_model_weights is None:
        raise RuntimeError("无有效训练窗口！请检查数据量或窗口参数")

    # 加载最优权重并保存
    final_model = build_unified_model(seq_length)
    final_model.set_weights(best_model_weights)
    final_model.save_weights(model_weights_path)

    # 保存窗口scaler和全局参考scaler（全局scaler用于预测时兜底）
    global_scaler = MinMaxScaler()
    global_scaler.fit(raw_data)
    with open(window_scalers_path, 'wb') as f:
        pickle.dump(window_scalers, f)
    with open(global_scaler_path, 'wb') as f:
        pickle.dump(global_scaler, f)

    print(f"\n✅ 局部标准化滑动窗口训练完成！")
    print(f"📊 最优验证损失：{best_val_loss:.4f}")
    print(f"💾 模型保存至：{model_weights_path}")
    print(f"💾 窗口scaler保存至：{window_scalers_path}")
    print(f"💾 全局参考scaler保存至：{global_scaler_path}")
    print(f"📋 共训练{len(window_scalers)}个有效窗口")

    return final_model, window_scalers, global_scaler

def predict_with_sliding_window_fixed(dataB, seq_length=32, model_weights_path='./lstm_model_weights.h5',
                                      scaler_path='./scaler_piezo.pkl', future_steps=16, target_total_points=None):
    """
    保留的预测函数（修复重复加载模型问题）
    """
    # 加载Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 加载模型（仅用统一结构+权重加载，不重复load_model）
    model = build_unified_model(seq_length)
    model.load_weights(model_weights_path)
    print(f"✅ 成功加载模型权重：{model_weights_path}")

    # 数据标准化
    col_name = 'Voltage' if 'Voltage' in dataB.columns else 'CH1V'
    dataB_scaled = scaler.transform(dataB[[col_name]])

    # 构建测试集窗口
    X_test = []
    for i in range(len(dataB_scaled) - seq_length):
        X_test.append(dataB_scaled[i:i + seq_length, 0])
    X_test = np.array(X_test).reshape(-1, seq_length, 1)

    # 计算预测段数
    max_possible_steps = len(X_test)
    if target_total_points is not None:
        total_steps = int(np.ceil(target_total_points / future_steps))
    else:
        total_steps = max_possible_steps
    total_steps = min(total_steps, max_possible_steps)
    if total_steps == 0:
        raise ValueError(f"数据B长度不足！X_test仅{len(X_test)}个样本")

    # 预测逻辑
    all_predicted_data = []
    time_list = []
    for step in range(total_steps):
        start_index = step * future_steps
        if start_index >= len(X_test):
            break
        last_sequence = X_test[start_index].reshape(1, seq_length, 1)

        predicted_data = []
        for i in range(future_steps):
            predicted_value = model.predict(last_sequence, verbose=0)
            predicted_data.append(predicted_value[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = predicted_value[0, 0]

        all_predicted_data.extend(predicted_data)
        start_point = start_index + seq_length
        time_list.extend([start_point + i for i in range(len(predicted_data))])

    # 逆标准化
    all_predicted_data = np.array(all_predicted_data).reshape(-1, 1)
    all_predicted_data_inversed = scaler.inverse_transform(all_predicted_data)
    time_data = np.array(time_list)

    print(f"✅ 预测完成：共分{total_steps}段，单次预测{future_steps}点，总预测{len(all_predicted_data)}点")
    return time_data, all_predicted_data_inversed


def predict_stepped_window_fast(dataB, seq_length=32, model_weights_path='./lstm_model_weights.h5',
                                scaler_path='./scaler_piezo.pkl', predict_step=32, target_total_points=None):
    """
    你实际调用的预测函数：
    1. 复用统一模型结构
    2. 关闭自回归，用真实窗口滚动
    3. 保留关键调试打印
    """
    # 1. 加载Scaler并验证范围
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"🔍 Scaler范围验证：min={scaler.data_min_[0]}, max={scaler.data_max_[0]}")

    # 2. 加载模型（用统一结构，确保和训练一致）
    model = build_unified_model(seq_length)
    model.load_weights(model_weights_path)
    tf.config.experimental.sync_to_device = False
    print(f"✅ 成功加载模型权重：{model_weights_path}")

    # 3. 数据标准化
    col_name = 'Voltage' if 'Voltage' in dataB.columns else 'CH1V'
    data_scaled = scaler.transform(dataB[[col_name]])
    data_length = len(data_scaled)
    print(f"🔍 输入数据标准化后范围：min={data_scaled.min()}, max={data_scaled.max()}")

    # 4. 确定目标预测量
    if target_total_points is None:
        target_total_points = data_length
    target_total_points = min(target_total_points, data_length - seq_length)

    # 5. 核心预测逻辑（真实窗口滚动，不自回归）
    all_pred_time = []
    all_pred_data = []
    current_start = 0

    while len(all_pred_data) < target_total_points:
        if current_start + seq_length + predict_step > data_length:
            break

        # 取真实数据窗口
        current_window = data_scaled[current_start:current_start + seq_length, 0]
        current_window = current_window.reshape(1, seq_length, 1)

        # 单次预测predict_step个点（仅用真实窗口）
        predicted_data = []
        for i in range(predict_step):
            pred = model.predict(current_window, verbose=0)[0, 0]
            predicted_data.append(pred)

        # 计算预测点序号
        pred_start = current_start + seq_length
        pred_time = [pred_start + i for i in range(predict_step)]

        # 累加结果
        all_pred_data.extend(predicted_data)
        all_pred_time.extend(pred_time)

        # 步进
        current_start += predict_step

        progress = min(len(all_pred_data) / target_total_points * 100, 100)
        print(f"预测进度：{progress:.1f}% | 已预测：{len(all_pred_data)}/{target_total_points} 点", end='\r')

    # 6. 截断+逆标准化
    all_pred_data = all_pred_data[:target_total_points]
    all_pred_time = all_pred_time[:target_total_points]
    print(f"\n🔍 模型预测值范围（标准化后）：min={np.min(all_pred_data)}, max={np.max(all_pred_data)}")

    pred_data_scaled = np.array(all_pred_data).reshape(-1, 1)
    pred_data_inversed = scaler.inverse_transform(pred_data_scaled)
    print(f"🔍 逆标准化后预测值范围：min={pred_data_inversed.min()}, max={pred_data_inversed.max()}")

    pred_time = np.array(all_pred_time)

    print(f"\n✅ 步进式预测完成：总预测{len(pred_data_inversed)}点，单次窗口{seq_length}，单次预测{predict_step}点")
    return pred_time, pred_data_inversed


# 核心：复刻旧代码的预测逻辑（自回归+小步预测）
def predict_old(dataB, seq_length=32, save_path='./', ratio=2):
    """
    预测函数：完全对齐旧代码的核心思路（自回归+小步预测），保留新代码的接口规范
    :param dataB: 输入数据（DataFrame，含Voltage列）
    :param seq_length: 窗口大小（默认32，对齐旧代码）
    :param save_path: 保存路径（默认当前目录）
    :param ratio: 步长比例（默认2 → future_steps=16，对齐旧代码）
    :return: 预测点时间索引、逆标准化后的预测值
    """
    # 1. 加载Scaler和模型（新代码的接口逻辑）
    scaler_path = os.path.join(save_path, 'scaler_piezo.pkl')
    model_weights_path = os.path.join(save_path, 'lstm_model_weights.h5')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model = build_unified_model(seq_length)
    model.load_weights(model_weights_path)
    print(f"✅ 成功加载模型权重：{model_weights_path}")

    # 2. 数据标准化（对齐旧代码）
    data_scaled = scaler.transform(dataB[['Voltage']])
    X, y = create_sequences(data_scaled, seq_length)

    # 3. 旧代码的80-20划分：只预测测试集
    split_index = int(len(X) * 0.8)
    X_test = X[split_index:]
    y_test = y[split_index:]

    # 4. 旧代码的核心：小步预测（future_steps=seq_length//ratio）
    future_steps = seq_length // ratio
    total_steps = len(y_test) // future_steps
    all_pred_data = []

    # 5. 旧代码的核心：自回归预测（预测值塞回窗口）
    for step in range(total_steps):
        start_index = step * future_steps
        if start_index >= len(X_test):
            break

        # 取测试集真实窗口
        last_sequence = X_test[start_index].reshape(1, seq_length, 1)
        predicted_data = []

        # 自回归：预测一个点，塞回窗口，再预测下一个
        for i in range(future_steps):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            predicted_data.append(pred)
            # 关键：np.roll滚动窗口，把预测值塞回（旧代码核心）
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred

        all_pred_data.extend(predicted_data)

    # 6. 逆标准化（新代码的处理逻辑）
    all_pred_data = np.array(all_pred_data).reshape(-1, 1)
    pred_data_inversed = scaler.inverse_transform(all_pred_data)

    # 7. 预测点时间索引（对齐新代码的接口）
    pred_time = np.arange(split_index + seq_length, split_index + seq_length + len(pred_data_inversed))

    # 8. 保留新代码的输出（仅核心输出，不加额外功能）
    print(f"\n✅ predict_old预测完成：")
    print(f"   - 窗口大小：{seq_length} | 单次预测步长：{future_steps}")
    print(f"   - 预测值范围：{pred_data_inversed.min():.4f} ~ {pred_data_inversed.max():.4f}")
    print(f"   - 总预测点数：{len(pred_data_inversed)}")

    return pred_time, pred_data_inversed


# ========== 适配局部标准化的predict_old函数 ==========
def predict_old_local_scaler(dataB, seq_length=32, save_path='./', ratio=2):
    """
    适配局部窗口标准化的预测函数（自回归+小步预测）
    :param dataB: 输入数据（DataFrame，含Voltage列）
    :param seq_length: 窗口大小
    :param save_path: 模型/scaler保存路径
    :param ratio: 步长比例（默认2 → future_steps=16）
    :return: 预测点时间索引、逆标准化后的预测值
    """
    # 1. 加载模型和scaler（适配局部标准化的训练函数输出）
    model_weights_path = os.path.join(save_path, 'lstm_model_weights_local.h5')
    window_scalers_path = os.path.join(save_path, 'window_scalers.pkl')
    global_scaler_path = os.path.join(save_path, 'global_scaler_piezo.pkl')

    # 加载模型
    model = build_unified_model(seq_length)
    model.load_weights(model_weights_path)
    print(f"✅ 成功加载模型权重：{model_weights_path}")

    # 加载窗口scaler和全局参考scaler
    with open(window_scalers_path, 'rb') as f:
        window_scalers = pickle.load(f)
    with open(global_scaler_path, 'rb') as f:
        global_scaler = pickle.load(f)
    print(f"✅ 加载{len(window_scalers)}个窗口的局部scaler")

    # 2. 80-20划分（用原始数据，不做全局标准化）
    raw_data = dataB[['Voltage']].values
    total_samples_raw = len(raw_data) - seq_length
    split_index = int(total_samples_raw * 0.8)
    X_test_raw = raw_data[split_index: split_index + total_samples_raw - split_index + seq_length]

    # 3. 旧代码核心：小步预测参数
    future_steps = seq_length // ratio
    total_steps = (total_samples_raw - split_index) // future_steps
    all_pred_data = []
    pred_time_list = []

    # 4. 自回归预测（核心逻辑，适配局部scaler）
    for step in range(total_steps):
        start_index = split_index + step * future_steps
        if start_index + seq_length > len(raw_data):
            break

        # 找到当前预测窗口所属的局部scaler
        # 匹配规则：找到最接近当前start_index的窗口scaler
        window_starts = sorted(window_scalers.keys())
        target_window_start = None
        for ws in window_starts:
            if ws <= start_index < ws + int(total_samples_raw * 0.1):  # 0.1是训练时的window_ratio
                target_window_start = ws
                break
        # 如果没找到匹配的窗口scaler，用全局scaler兜底
        if target_window_start is None:
            target_scaler = global_scaler
            print(f"⚠️ 预测窗口{start_index}未匹配到局部scaler，使用全局scaler")
        else:
            target_scaler = window_scalers[target_window_start]
            print(f"✅ 预测窗口{start_index}匹配到局部scaler（窗口起始：{target_window_start}）")

        # 取当前窗口的原始数据，做局部标准化
        current_raw_window = raw_data[start_index: start_index + seq_length]
        current_scaled_window = target_scaler.transform(current_raw_window)
        last_sequence = current_scaled_window.reshape(1, seq_length, 1)
        predicted_data_scaled = []

        # 自回归预测（小步）
        for i in range(future_steps):
            pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            predicted_data_scaled.append(pred_scaled)
            # 滚动窗口：预测值塞回（用标准化后的值）
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred_scaled

        # 逆标准化：用匹配的局部scaler还原幅值
        predicted_data_scaled = np.array(predicted_data_scaled).reshape(-1, 1)
        predicted_data_inversed = target_scaler.inverse_transform(predicted_data_scaled)
        all_pred_data.extend(predicted_data_inversed.flatten().tolist())

        # 记录预测时间索引
        pred_start = start_index + seq_length
        pred_time_list.extend(range(pred_start, pred_start + future_steps))

    # 5. 整理输出
    all_pred_data = np.array(all_pred_data).reshape(-1, 1)
    pred_time = np.array(pred_time_list[:len(all_pred_data)])  # 对齐长度

    print(f"\n✅ predict_old（局部标准化）预测完成：")
    print(f"   - 窗口大小：{seq_length} | 单次预测步长：{future_steps}")
    print(f"   - 预测值范围：{all_pred_data.min():.4f} ~ {all_pred_data.max():.4f}")
    print(f"   - 总预测点数：{len(all_pred_data)}")

    return pred_time, all_pred_data



def plot_predicted_data(time_data, predicted_data_inversed):
    # 绘制预测结果
    plt.plot(time_data, predicted_data_inversed, label="Predicted Voltage")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend()
    plt.show()


def plot_double_figure(true_data, time_true, pred_data, time_pred):
    """绘制真实/预测对比图"""
    # 图1：真实 vs 预测
    plt.figure(figsize=(12, 5))
    plt.plot(time_true, true_data['Voltage'].values, label='真实电压', alpha=0.7, color='blue')
    plt.plot(time_pred, pred_data, label='预测电压', alpha=0.7, color='red')
    plt.xlabel('采样点序号')
    plt.ylabel('电压 (V)')
    plt.title('真实电压 vs 预测电压对比（采样点序号）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 图2：仅预测电压
    plt.figure(figsize=(12, 5))
    plt.plot(time_pred, pred_data, color='red', label='预测电压')
    plt.xlabel('采样点序号')
    plt.ylabel('电压 (V)')
    plt.title('预测电压随采样点变化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()