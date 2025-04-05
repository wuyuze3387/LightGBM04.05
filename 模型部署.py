import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文和负号，这里以微软雅黑为例
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 加载模型
model_path = "LGBMRegressor.pkl"
model = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="轻量级梯度提升回归模型预测与 SHAP 可视化", page_icon="💕👩‍⚕️🏥")
st.title("💕👩‍⚕️🏥 轻量级梯度提升回归模型预测与 SHAP 可视化")
st.write("通过输入所有变量的值进行单个样本分娩心理创伤的风险预测，可以得到该样本罹患分娩心理创伤的概率，并结合 SHAP 瀑布图分析结果，有助于临床医护人员了解具体的风险因素和保护因素。")

# 特征范围定义
feature_ranges = {
    "年龄": {"type": "numerical", "min": 18, "max": 42, "default": 18},
    "体重": {"type": "numerical", "min": 52, "max": 91, "default": 52},
    "居住地": {"type": "categorical", "options": [1, 2]},
    "婚姻状况": {"type": "categorical", "options": [1, 2]},
    "就业情况": {"type": "categorical", "options": [1, 2]},
    "学历": {"type": "categorical", "options": [1, 2, 3, 4]},
    "医疗费用支付方式": {"type": "categorical", "options": [1, 2, 3]},
    "怀孕次数": {"type": "numerical", "min": 1, "max": 8, "default": 1},
    "分娩次数": {"type": "numerical", "min": 1, "max": 4, "default": 1},
    "分娩方式": {"type": "categorical", "options": [1, 2, 3]},
    "不良孕产史": {"type": "categorical", "options": [1, 2]},
    "终止妊娠经历": {"type": "categorical", "options": [1, 2]},
    "妊娠周数": {"type": "numerical", "min": 29, "max": 44, "default": 29},
    "妊娠合并症": {"type": "categorical", "options": [1, 2]},
    "妊娠并发症": {"type": "categorical", "options": [1, 2]},
    "喂养方式": {"type": "categorical", "options": [1, 2, 3]},
    "新生儿是否有出生缺陷或疾病": {"type": "categorical", "options": [1, 2]},
    "家庭人均月收入": {"type": "categorical", "options": [1, 2]},
    "使用无痛分娩技术": {"type": "categorical", "options": [1, 2]},
    "产时疼痛": {"type": "numerical", "min": 0, "max": 10, "default": 0},
    "产后疼痛": {"type": "numerical", "min": 1, "max": 9, "default": 1},
    "产后照顾婴儿方式": {"type": "categorical", "options": [1, 2, 3, 4, 5]},
    "近1月睡眠质量": {"type": "categorical", "options": [1, 2, 3, 4]},
    "近1月夜间睡眠时长": {"type": "numerical", "min": 3, "max": 11, "default": 3},
    "近1月困倦程度": {"type": "categorical", "options": [1, 2, 3, 4]},
    "孕期体育活动等级": {"type": "categorical", "options": [1, 2, 3, 4]},
    "抑郁": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "焦虑": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "侵入性反刍性沉思": {"type": "numerical", "min": 0, "max": 30, "default": 0},
    "目的性反刍性沉思": {"type": "numerical", "min": 0, "max": 28, "default": 0},
    "心理弹性": {"type": "numerical", "min": 6, "max": 30, "default": 6},
    "家庭支持": {"type": "numerical", "min": 0, "max": 10, "default": 0},
}
# 动态生成输入项
st.sidebar.header("变量输入区域")
st.sidebar.write("请输入变量值：")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("预测"):
    # 模型预测
    predicted_value = model.predict(features)[0]
    st.write(f"Predicted 分娩心理创伤 score: {predicted_value:.2f}%")

    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 获取基础值和第一个样本的 SHAP 值
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[0]  # 如果是列表，取第一个元素
    else:
        base_value = explainer.expected_value  # 如果是单个值，直接使用

    shap_values_sample = shap_values[0]

    # 定义特征名称和其对应的值
    features_with_values = np.array([
        f"Age={feature_values[0]}",
        f"Weight={feature_values[1]}",
        f"Residence={feature_values[2]}",
        f"Marriage={feature_values[3]}",
        f"Employment={feature_values[4]}",
        f"Education={feature_values[5]}",
        f"Insurance={feature_values[6]}",
        f"Pregnancies={feature_values[7]}",
        f"Deliveries={feature_values[8]}",
        f"Delivery Method={feature_values[9]}",
        f"Adverse Pregnancy History={feature_values[10]}",
        f"Terminated Pregnancy={feature_values[11]}",
        f"Pregnancy Weeks={feature_values[12]}",
        f"Comorbidities={feature_values[13]}",
        f"Complications={feature_values[14]}",
        f"Feeding={feature_values[15]}",
        f"Newborn Defects={feature_values[16]}",
        f"Monthly Income Per Capita={feature_values[17]}",
        f"Painless Childbirth={feature_values[18]}",
        f"Intra_pain={feature_values[19]}",
        f"Post_Pain={feature_values[20]}",
        f"Care Methods={feature_values[21]}",
        f"Sleep Quality={feature_values[22]}",
        f"Sleep Time={feature_values[23]}",
        f"Fatigue={feature_values[24]}",
        f"Activity={feature_values[25]}",
        f"Dep={feature_values[26]}",
        f"Anx={feature_values[27]}",
        f"Intrusive Rumination={feature_values[28]}",
        f"Purposeful Rumination={feature_values[29]}",
        f"Resilience={feature_values[30]}",
        f"Fami_Supp={feature_values[31]}"
    ])

    # 创建SHAP瀑布图，确保中文显示
    plt.figure(figsize=(20, 6))  # 设置图形尺寸为20x6英寸
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_sample,
            base_values=base_value,
            data=features[0],
            feature_names=list(feature_ranges.keys())
        ),
        max_display=10  # 限制显示的特征数量
    )

    # 保存SHAP瀑布图并展示
    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=600)
    st.image("shap_waterfall_plot.png")
