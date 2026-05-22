
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 設定頁面資訊
st.set_page_config(page_title="玻璃物性預測系統", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("optimized_glass_model.joblib")

def predict_properties_with_correction(composition_dict, model_package):
    f_cols = model_package["input_features"]
    x = pd.DataFrame([{c: 0.0 for c in f_cols}])
    for k, v in composition_dict.items():
        col = k if k.endswith("_mass_pct") else k + "_mass_pct"
        if col in x.columns:
            x.loc[0, col] = float(v)

    # 歸一化至 100%
    total = x[f_cols].sum(axis=1).iloc[0]
    if total > 0:
        x[f_cols] = x[f_cols].div(total, axis=0) * 100

    # 執行模型預測
    preds = {t: model_package["models"][t].predict(x[f_cols])[0] for t in model_package["models"]}

    # CTE 偏差修正邏輯
    original_cte = preds.get("cte_1e-6_per_C", 0)
    if original_cte < 3.5:
        b2o3_content = composition_dict.get("B2O3", 0)
        correction = 0.15 + (0.03 * b2o3_content)
        preds["cte_1e-6_per_C"] = max(original_cte - correction, 0.1)
    
    return preds

# --- UI 介面 ---
st.title("🔬 玻璃物理性質 AI 預測系統")
st.markdown("--- ")

try:
    model_package = load_model()
    
    # --- 上方：配方輸入區 ---
    st.subheader("🧪 1. 設定玻璃配方 (mass %)")
    
    # 取得模型支援的所有組分清單
    all_available_oxides = sorted([c.replace("_mass_pct", "") for c in model_package["input_features"]])
    default_oxides = ["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"]

    # 讓使用者動態選擇要顯示的組分
    selected_oxides = st.multiselect(
        "選擇要輸入的氧化物組分：", 
        options=all_available_oxides, 
        default=default_oxides
    )

    # 使用 Columns 佈局顯示輸入框
    input_comps = {ox: 0.0 for ox in all_available_oxides}
    cols = st.columns(min(len(selected_oxides), 6)) 
    for i, oxide in enumerate(selected_oxides):
        with cols[i % 6]:
            input_comps[oxide] = st.number_input(f"{oxide}", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"in_{oxide}")

    st.markdown("--- ")

    # --- 下方：預測按鈕與結果 ---
    if st.button("🚀 執行計算並預測物性", use_container_width=True):
        results = predict_properties_with_correction(input_comps, model_package)
        
        st.subheader("📊 2. 預測結果")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("CTE (熱膨脹係數)", f"{results['cte_1e-6_per_C']:.4f}", "10^-6/°C")
        with res_col2:
            st.metric("Young's Modulus (楊氏模量)", f"{results['young_modulus_GPa']:.2f}", "GPa")
        with res_col3:
            st.metric("Viscosity Temp (T at 10^3)", f"{results['T_at_1E3_dPas_C']:.2f}", "°C")
        
        # 顯示當前歸一化後的配方百分比供參考
        with st.expander("查看歸一化後的配方細節"):
            current_df = pd.DataFrame([input_comps])
            non_zero = current_df.loc[:, (current_df != 0).any(axis=0)]
            st.write(non_zero)

except Exception as e:
    st.error(f"發生錯誤: {e}")
