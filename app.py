
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sklearn

st.set_page_config(page_title="玻璃預測系統", layout="centered")

st.title("🔬 玻璃物理性質 AI 預測系統")

# 狀態檢查清單
with st.sidebar:
    st.header("系統狀態檢查")
    files_to_check = ["optimized_glass_model.joblib", "sciglass_database.csv", "interglad_database.csv"]
    env_ok = True
    for f in files_to_check:
        if os.path.exists(f):
            st.success(f"✅ {f} 已就緒")
        else:
            st.error(f"❌ 缺少 {f}")
            env_ok = False
    st.info(f"sklearn version: {sklearn.__version__}")

if not env_ok:
    st.warning("請確保所有必要檔案都已上傳到 GitHub 根目錄。")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load("optimized_glass_model.joblib")

# 使用 st.spinner 處理耗時任務
with st.spinner('正在載入 AI 模型，請稍候...'):
    try:
        model_pkg = get_model()
        st.sidebar.success("🔥 模型載入成功")
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        st.stop()

# 預測邏輯
def predict(comp, pkg):
    f_cols = pkg["input_features"]
    x = pd.DataFrame([{c: 0.0 for c in f_cols}])
    for k, v in comp.items():
        col = k if k.endswith("_mass_pct") else k + "_mass_pct"
        if col in x.columns: x.loc[0, col] = float(v)
    # 歸一化
    total = x[f_cols].sum(axis=1).iloc[0]
    if total > 0: x[f_cols] = x[f_cols].div(total, axis=0) * 100
    
    preds = {t: pkg["models"][t].predict(x[f_cols])[0] for t in pkg["models"]}
    # CTE 偏差修正
    if preds.get("cte_1e-6_per_C", 10) < 3.5:
        corr = 0.15 + (0.03 * comp.get("B2O3", 0))
        preds["cte_1e-6_per_C"] = max(preds["cte_1e-6_per_C"] - corr, 0.1)
    return preds

# 主 UI
st.subheader("1. 輸入玻璃配方 (mass %)")
all_oxides = sorted([c.replace("_mass_pct", "") for c in model_pkg["input_features"]])
selected = st.multiselect("選擇組分", all_oxides, default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"])

input_data = {}
cols = st.columns(3)
for i, ox in enumerate(selected):
    with cols[i % 3]:
        input_data[ox] = st.number_input(f"{ox}", value=0.0, step=0.1)

if st.button("🚀 開始預測", use_container_width=True):
    res = predict(input_data, model_pkg)
    st.divider()
    st.subheader("2. 預測結果")
    r1, r2, r3 = st.columns(3)
    r1.metric("CTE", f"{res['cte_1e-6_per_C']:.4f}")
    r2.metric("Young's Modulus", f"{res['young_modulus_GPa']:.2f} GPa")
    r3.metric("Viscosity (T10^3)", f"{res['T_at_1E3_dPas_C']:.1f} °C")
