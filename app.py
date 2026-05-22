
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
import os

st.set_page_config(page_title="玻璃物性預測系統", layout="wide")

# --- 部署環境檢查 ---
def check_env():
    missing_files = []
    required_files = ["optimized_glass_model.joblib", "sciglass_database.csv", "interglad_database.csv"]
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    return missing_files

st.title("🔬 玻璃物理性質 AI 預測系統")

missing = check_env()
if missing:
    st.error(f"❌ 部署檔案缺失: {', '.join(missing)}")
    st.warning("請確保已將模型檔與 CSV 資料庫上傳至 GitHub 同一目錄下。")
    st.stop() # 停止執行後續程式碼

@st.cache_resource
def load_model():
    try:
        return joblib.load("optimized_glass_model.joblib")
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        st.info(f"當前環境 sklearn 版本: {sklearn.__version__}")
        return None

def predict_properties_with_correction(composition_dict, model_package):
    f_cols = model_package["input_features"]
    x = pd.DataFrame([{c: 0.0 for c in f_cols}])
    for k, v in composition_dict.items():
        col = k if k.endswith("_mass_pct") else k + "_mass_pct"
        if col in x.columns:
            x.loc[0, col] = float(v)
    total = x[f_cols].sum(axis=1).iloc[0]
    if total > 0:
        x[f_cols] = x[f_cols].div(total, axis=0) * 100
    preds = {t: model_package["models"][t].predict(x[f_cols])[0] for t in model_package["models"]}
    original_cte = preds.get("cte_1e-6_per_C", 0)
    if original_cte < 3.5:
        b2o3_content = composition_dict.get("B2O3", 0)
        correction = 0.15 + (0.03 * b2o3_content)
        preds["cte_1e-6_per_C"] = max(original_cte - correction, 0.1)
    return preds

model_package = load_model()
if model_package:
    st.success("✅ 系統初始化成功")
    st.subheader("🧪 1. 設定玻璃配方 (mass %)")
    all_available_oxides = sorted([c.replace("_mass_pct", "") for c in model_package["input_features"]])
    selected_oxides = st.multiselect("選擇要輸入的氧化物組分：", options=all_available_oxides, default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"])
    input_comps = {ox: 0.0 for ox in all_available_oxides}
    cols = st.columns(min(len(selected_oxides), 6))
    for i, oxide in enumerate(selected_oxides):
        with cols[i % 6]:
            input_comps[oxide] = st.number_input(f"{oxide}", min_value=0.0, max_value=100.0, step=0.1, key=f"in_{oxide}")

    if st.button("🚀 執行計算", use_container_width=True):
        results = predict_properties_with_correction(input_comps, model_package)
        st.subheader("📊 預測結果")
        c1, c2, c3 = st.columns(3)
        c1.metric("CTE", f"{results['cte_1e-6_per_C']:.4f}")
        c2.metric("Young's Modulus", f"{results['young_modulus_GPa']:.2f}")
        c3.metric("Viscosity Temp", f"{results['T_at_1E3_dPas_C']:.2f}")
