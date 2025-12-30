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
from tensorflow.python.keras.saving.save import load_model
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


def ensemble_loss(y_true, y_pred, a=0.3, b=0.7):
    # a=0.3 b=0.7 55开时候应该最好，强于37，强于64
    # 现在是64最好，且要每次不用模型预测
    # 保留64，在数据集20-18-60-32.5上效果很差
    # 定义集成式损失函数，根据需要调整权重 a 和 b
    loss1 = tf.square(y_true - y_pred)  # 第一个损失函数
    loss2 = tf.abs(y_true - y_pred)     # 第二个损失函数
    return a * loss1 + b * loss2        # 加权求和


# 新增：全局标准化函数（训练和预测用同一个scaler，避免分布偏移）
def create_scaler(data):
    """基于训练数据创建scaler，供后续预测使用"""
    scaler = MinMaxScaler()
    scaler.fit(data[['Voltage']])
    return scaler


# ========== 3. 构建数据集（不变） ==========
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


def train_lstm_attention_model(preprocessed_df, seq_length=32, save_path='./',
                               scaler=None, roll_window_ratio=0.2, roll_step_ratio=0.1):
    """
    带滚动验证的LSTM-Attention训练（适配时序数据，避免分布偏移）
    :param preprocessed_df: 预处理后的DataFrame（Voltage列）
    :param seq_length: 时间步
    :param save_path: 模型保存路径
    :param scaler: 外部scaler
    :param roll_window_ratio: 每个验证窗口占总数据的比例（默认20%）
    :param roll_step_ratio: 滚动步长占总数据的比例（默认10%）
    :return: 训练好的模型、scaler
    """
    # 1. 标准化
    if scaler is None:
        scaler = create_scaler(preprocessed_df)
    data_scaled = scaler.transform(preprocessed_df[['Voltage']])

    # 2. 构建完整序列数据集
    X_all, y_all = create_sequences(data_scaled, seq_length)
    total_samples = len(X_all)

    # 3. 滚动验证参数计算
    val_window_size = int(total_samples * roll_window_ratio)  # 每个验证窗口大小（20%）
    roll_step = int(total_samples * roll_step_ratio)  # 滚动步长（10%）
    if val_window_size == 0 or roll_step == 0:
        raise ValueError("数据量过小，无法进行滚动验证！请增大数据量或调整窗口/步长比例")

    # 4. 模型结构（和你原代码一致）
    def build_model():
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

    # 5. 模型路径
    model_filename = 'lstm_attention_piezo_model.h5'
    scaler_filename = 'scaler_piezo.pkl'
    model_path = os.path.join(save_path, model_filename)
    scaler_path = os.path.join(save_path, scaler_filename)

    # 6. 加载/初始化模型
    if os.path.exists(model_path):
        print(f"加载已有模型：{model_path}")
        model = load_model(
            model_path,
            custom_objects={'ensemble_loss': ensemble_loss, 'Attention': Attention}
        )
    else:
        print("初始化新模型，开始滚动训练...")
        model = build_model()

    # 7. 滚动验证训练（核心逻辑）
    start_idx = 0
    val_loss_list = []  # 记录每个窗口的验证损失
    while start_idx + val_window_size <= total_samples:
        # 划分当前训练/验证窗口（严格按时间顺序，不打乱）
        train_end_idx = start_idx
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_window_size

        X_train = X_all[:train_end_idx] if train_end_idx > 0 else X_all[:val_start_idx]
        y_train = y_all[:train_end_idx] if train_end_idx > 0 else y_all[:val_start_idx]
        X_val = X_all[val_start_idx:val_end_idx]
        y_val = y_all[val_start_idx:val_end_idx]

        # 跳过样本不足的情况
        if len(X_train) < 100 or len(X_val) < 50:  # 最小样本阈值
            start_idx += roll_step
            continue

        # 训练当前窗口（增量训练，贴合时序分布）
        print(f"\n=== 滚动窗口 {start_idx // roll_step + 1} ===")
        print(f"训练集：0 ~ {train_end_idx if train_end_idx > 0 else val_start_idx} 样本")
        print(f"验证集：{val_start_idx} ~ {val_end_idx} 样本")

        history = model.fit(
            X_train, y_train,
            epochs=2,  # 每个窗口少量epochs，避免过拟合
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=False  # 时序数据绝对不能shuffle！
        )

        # 记录验证损失
        val_loss = history.history['val_loss'][-1]
        val_loss_list.append(val_loss)
        print(f"当前窗口验证损失：{val_loss:.4f}")

        # 滚动到下一个窗口
        start_idx += roll_step

    # 8. 保存最终模型和scaler
    if not os.path.exists(model_path):
        model.save(model_path)
        print(f"\n滚动训练完成，模型保存至：{model_path}")
    print(f"平均验证损失：{np.mean(val_loss_list):.4f}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler保存至：{scaler_path}")

    return model, scaler


def predict_with_sliding_window_fixed(dataB, seq_length=32, model_path='./lstm_attention_model_full_dataA.h5',
                                      scaler_path='./scaler.pkl', future_steps=16, target_total_points=None):
    """
    修正后的预测逻辑：对齐代码1，分段预测+真实数据重置窗口（动态适配长度）
    :param dataB: 输入数据（DataFrame，含CH1V/Voltage列 + Time(s)列）
    :param seq_length: 时间步长（窗口大小）
    :param model_path: 模型路径
    :param scaler_path: 标准化器路径
    :param future_steps: 单次预测长度（每段预测的点数，保持你的16不变）
    :param target_total_points: 目标总预测点数（可选，默认预测到数据B的最大长度）
    :return: 预测时间序列、逆标准化后的预测电压值
    """
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 适配列名：兼容CH1V（你的旧数据）和Voltage（咱们的新数据）
    col_name = 'CH1V' if 'CH1V' in dataB.columns else 'Voltage'

    # 移除清洗（完全保留你的逻辑）
    # dataB = dataB[(dataB[col_name] > dataB[col_name].quantile(0.01)) &
    #               (dataB[col_name] < dataB[col_name].quantile(0.99))]

    # 标准化（适配动态列名）
    dataB_scaled = scaler.transform(dataB[[col_name]])

    # 构建测试集窗口（完全保留你的逻辑）
    X_test = []
    for i in range(len(dataB_scaled) - seq_length):
        X_test.append(dataB_scaled[i:i + seq_length, 0])
    X_test = np.array(X_test).reshape(-1, seq_length, 1)

    # ========== 核心优化：动态计算总段数 ==========
    # 1. 计算X_test能支持的最大段数（避免索引越界）
    max_possible_steps = len(X_test)  # 每段至少用1个X_test样本初始化窗口
    # 2. 若指定了目标总点数，计算需要的段数；否则用最大可能段数
    if target_total_points is not None:
        # 向上取整：确保总点数≥目标值（如目标1000，16/段 → 63段=1008点）
        total_steps = int(np.ceil(target_total_points / future_steps))
    else:
        # 无目标时，预测到X_test的最大长度（每段用1个初始化样本）
        total_steps = max_possible_steps

    # 安全校验：段数不能超过X_test的最大支持数
    total_steps = min(total_steps, max_possible_steps)
    if total_steps == 0:
        raise ValueError(f"数据B长度不足！X_test仅{len(X_test)}个样本，无法完成至少1段预测")

    # 加载模型（完全保留你的逻辑）
    model = tf.keras.models.load_model(model_path, custom_objects={'ensemble_loss': ensemble_loss})

    all_predicted_data = []
    time_list = []
    for step in range(total_steps):
        # 1. 用数据B的真实数据初始化窗口（核心逻辑完全保留）
        start_index = step * future_steps  # 按段数步进，和你原逻辑一致
        if start_index >= len(X_test):
            break
        last_sequence = X_test[start_index].reshape(1, seq_length, 1)

        # 2. 单次预测future_steps个点（完全保留你的滚动窗口逻辑）
        predicted_data = []
        for i in range(future_steps):
            predicted_value = model.predict(last_sequence, verbose=0)
            predicted_data.append(predicted_value[0, 0])
            # 滚动窗口（仅在当前段内）
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = predicted_value[0, 0]

        # 3. 收集预测结果和时间（完全保留你的时间计算逻辑）
        all_predicted_data.extend(predicted_data)
        # 生成对应时间（适配动态列名）
        start_time = dataB['Time(s)'].iloc[start_index + seq_length]
        time_interval = dataB['Time(s)'].iloc[1] - dataB['Time(s)'].iloc[0]
        time_list.extend([start_time + i * time_interval for i in range(len(predicted_data))])

    # 逆标准化（完全保留）
    all_predicted_data = np.array(all_predicted_data).reshape(-1, 1)
    all_predicted_data_inversed = scaler.inverse_transform(all_predicted_data)
    time_data = np.array(time_list)

    # 输出预测信息（方便调试）
    print(f"✅ 预测完成：共分{total_steps}段，单次预测{future_steps}点，总预测{len(all_predicted_data)}点")
    return time_data, all_predicted_data_inversed



def plot_predicted_data(time_data, predicted_data_inversed):
    # 绘制预测结果
    plt.plot(time_data, predicted_data_inversed, label="Predicted Voltage")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    # plt.title("预测的电压随时间变化")
    plt.legend()
    plt.show()


# ========== 新增：你需要的双图对比绘制函数 ==========
def plot_double_figure(true_data, time_true, pred_data, time_pred):
    """
    绘制两张对比图：
    图1：真实数据 + 预测数据在同一张图对比
    图2：仅预测数据的独立展示图
    """
    # 图1：真实数据 vs 预测数据（对比图）
    plt.figure(figsize=(12, 5))
    plt.plot(time_true, true_data['Voltage'].values, label='真实电压', alpha=0.7, color='blue')
    plt.plot(time_pred, pred_data, label='预测电压', alpha=0.7, color='red')
    plt.xlabel('时间 (s)')
    plt.ylabel('电压 (V)')
    plt.title('真实电压 vs 预测电压对比')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 图2：仅预测数据（独立展示）
    plt.figure(figsize=(12, 5))
    plt.plot(time_pred, pred_data, color='red', label='预测电压')
    plt.xlabel('时间 (s)')
    plt.ylabel('电压 (V)')
    plt.title('预测电压随时间变化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# 主函数
def main():  # 旧有数据的实验
    # *******步骤1：训练部分
    file_nameA = 'RigolDS0.csv'

    dataA=extract_data_from_csv(file_nameA)
    scaler = train_lstm_model(dataA, seq_length=32, save_path='./')




    # *******步骤2：预测部分
    dataB = extract_data_from_csv('RigolDS1.csv')
    time_data, predicted_data_inversed = predict_with_sliding_window_fixed(dataB, seq_length=32,
                                                                     model_path='./lstm_attention_model_full_dataA.h5',
                                                                     scaler_path='./scaler.pkl',
                                                                     future_steps=16,
                                                                     total_steps=62)  # 62*16=992≈1000个点

    # 同时绘制真实数据和预测数据（方便对比）
    plt.figure(figsize=(12, 6))
    plt.plot(dataB['Time(s)'].values, dataB['CH1V'].values, label='TrueData', alpha=0.7)
    plt.plot(time_data, predicted_data_inversed, label='PredictData', color='red', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend()
    plt.show()


    # plot_predicted_data(dataB['Time(s)'].values, dataB['CH1V'].values)
    # plot_predicted_data(time_data, predicted_data_inversed)
    # save_to_csv(time_data, predicted_data_inversed,'1113test.csv')
    # train_lstm_model(dataA) # 训练模型
    # plot_voltage_time(dataA)
    # save_to_csv(time_data, voltage_data)



if __name__ == "__main__":
    main()
