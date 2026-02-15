from pathlib import Path
import glob

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title='MPN Risk Calculator', page_icon=':bar_chart:', layout='wide')

ARTIFACT_DIR = Path('saved_models')
PREP_PATH = ARTIFACT_DIR / 'preprocessing_artifacts.joblib'
SELECTOR_PATH = ARTIFACT_DIR / 'feature_selector_artifacts.joblib'
DEFAULT_MODEL_NAME = 'LR'
DECISION_THRESHOLD = 0.5
FEV1_ABS_MIN, FEV1_ABS_MAX = 0.1, 10.0
FEV1_PRED_PCT_MIN, FEV1_PRED_PCT_MAX = 0.0, 200.0
NODULE_DIAMETER_MIN, NODULE_DIAMETER_MAX = 0.50, 3.00


REQUIRED_FEATURES = [
    'FEV1_Abs',
    'FEV1_Pred_Pct',
    'Loc_RML',
    'Loc_LUL',
    'Loc_LLL',
    'Lobe_Segment',
    'Radiology_Feature',
    'Nodule_Diameter',
]

LOBE_SEGMENT_OPTIONS = {
    1: 'LUL apicoposterior',
    2: 'LUL anterior',
    3: 'LUL superior lingular',
    4: 'LUL inferior lingular',
    5: 'LLL superior',
    6: 'LLL anteromedial basal',
    7: 'LLL lateral basal',
    8: 'LLL posterior basal',
    9: 'RUL apical',
    10: 'RUL posterior',
    11: 'RUL anterior',
    12: 'RML lateral',
    13: 'RML medial',
    14: 'RLL superior',
    15: 'RLL medial basal',
    16: 'RLL anterior basal',
    17: 'RLL lateral basal',
    18: 'RLL posterior basal',
}

RADIOLOGY_FEATURE_OPTIONS = {
    1: 'Solid',
    2: 'Part-solid',
    3: 'Pure GGN',
}


@st.cache_resource
def load_artifacts():
    if not PREP_PATH.exists():
        raise FileNotFoundError(f'Missing file: {PREP_PATH}')
    if not SELECTOR_PATH.exists():
        raise FileNotFoundError(f'Missing file: {SELECTOR_PATH}')

    prep = joblib.load(PREP_PATH)
    selector = joblib.load(SELECTOR_PATH)

    model_paths = sorted(glob.glob(str(ARTIFACT_DIR / 'best_model_*.joblib')))
    if not model_paths:
        raise FileNotFoundError('No model file found in saved_models/best_model_*.joblib')

    models = {}
    for p in model_paths:
        name = Path(p).stem.replace('best_model_', '')
        models[name] = joblib.load(p)

    return prep, selector, models


def build_scaler_lookup(prep):
    scaler = prep.get('scaler')
    if scaler is None:
        return {}

    feature_names = list(getattr(scaler, 'feature_names_in_', []))
    if not feature_names:
        feature_names = prep.get('continuous_cols_before_drop', [])

    means = getattr(scaler, 'mean_', None)
    scales = getattr(scaler, 'scale_', None)
    if means is None or scales is None:
        return {}

    lookup = {}
    for idx, col in enumerate(feature_names):
        scale = scales[idx] if scales[idx] != 0 else 1.0
        lookup[col] = (means[idx], scale)
    return lookup


def to_model_frame(raw_inputs, selected_features, scaler_lookup):
    row = {}
    for col in selected_features:
        value = raw_inputs[col]
        if col in scaler_lookup:
            mean, scale = scaler_lookup[col]
            value = (float(value) - float(mean)) / float(scale)
        row[col] = value
    return pd.DataFrame([row], columns=selected_features)


st.title('MPN Prediction Calculator')

try:
    prep_artifacts, selector_artifacts, model_pool = load_artifacts()
except Exception as exc:
    st.error(f'Artifact loading failed: {exc}')
    st.stop()

selected_features = selector_artifacts.get('selected_features', [])
if selected_features != REQUIRED_FEATURES:
    st.warning(
        'Current selector feature list differs from the expected 8 features. '
        'The app will use selector order for prediction.'
    )

scaler_lookup = build_scaler_lookup(prep_artifacts)
feature_order = selected_features if selected_features else REQUIRED_FEATURES
if DEFAULT_MODEL_NAME not in model_pool:
    available = ', '.join(model_pool.keys())
    st.error(f'DEFAULT_MODEL_NAME="{DEFAULT_MODEL_NAME}" not found. Available models: {available}')
    st.stop()

fixed_model = model_pool[DEFAULT_MODEL_NAME]
left_col, right_col = st.columns([1.35, 1], gap='large')

with left_col:
    with st.form('predict_form'):
        st.subheader('Input Features')

        c1, c2 = st.columns(2)
        with c1:
            fev1_abs = st.number_input(
                'FEV1_Abs',
                min_value=FEV1_ABS_MIN,
                max_value=FEV1_ABS_MAX,
                value=2.5,
                step=0.01,
                format='%.2f'
            )
            fev1_pred_pct = st.number_input(
                'FEV1_Pred_Pct',
                min_value=FEV1_PRED_PCT_MIN,
                max_value=FEV1_PRED_PCT_MAX,
                value=85.0,
                step=0.1,
                format='%.2f'
            )
            loc_rml = st.number_input(
                'Loc_RML_Num',
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )
            loc_lul = st.number_input(
                'Loc_LUL_Num',
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

        with c2:
            loc_lll = st.number_input(
                'Loc_LLL_Num',
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )
            lobe_segment = st.selectbox(
                'Segment_location (Lobe_Segment)',
                options=list(LOBE_SEGMENT_OPTIONS.keys()),
                index=0,
                format_func=lambda x: LOBE_SEGMENT_OPTIONS[x]
            )
            radiology_feature = st.selectbox(
                'Radiology_Feature',
                options=list(RADIOLOGY_FEATURE_OPTIONS.keys()),
                index=0,
                format_func=lambda x: RADIOLOGY_FEATURE_OPTIONS[x]
            )
            nodule_diameter = st.number_input(
                'Nodule_Diameter',
                min_value=NODULE_DIAMETER_MIN,
                max_value=NODULE_DIAMETER_MAX,
                value=1.00,
                step=0.01,
                format='%.2f'
            )

        submit = st.form_submit_button('Predict')

    if submit:
        raw_inputs = {
            'FEV1_Abs': fev1_abs,
            'FEV1_Pred_Pct': fev1_pred_pct,
            'Loc_RML': loc_rml,
            'Loc_LUL': loc_lul,
            'Loc_LLL': loc_lll,
            'Lobe_Segment': float(lobe_segment),
            'Radiology_Feature': float(radiology_feature),
            'Nodule_Diameter': nodule_diameter,
        }

        missing = [f for f in feature_order if f not in raw_inputs]
        if missing:
            st.error(f'Missing required inputs: {missing}')
            st.stop()

        X = to_model_frame(raw_inputs, feature_order, scaler_lookup)
        if hasattr(fixed_model, 'predict_proba'):
            prob_pos = float(fixed_model.predict_proba(X)[0, 1])
        else:
            score = float(fixed_model.decision_function(X)[0])
            prob_pos = 1.0 / (1.0 + np.exp(-score))

        pred_label = int(prob_pos >= DECISION_THRESHOLD)
        st.session_state['result_prob_pos'] = prob_pos
        st.session_state['result_pred_text'] = 'IN' if pred_label == 1 else 'NIN'

with right_col:
    st.subheader('Prediction Result')
    if 'result_prob_pos' in st.session_state:
        st.metric('Positive Probability', f"{st.session_state['result_prob_pos']:.4f}")
        st.metric('Predicted Class', st.session_state['result_pred_text'])
    else:
        st.info('Submit input values on the left to generate a prediction.')

st.divider()
st.caption('Notes:')
st.caption('This tool is for research assistance only and should not be used as the sole basis for clinical diagnosis.')
