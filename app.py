import streamlit as st
import pandas as pd
import joblib
import traceback

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Success Predictor")

# =====================================================
# LOAD MODEL
# =====================================================
try:
    bundle           = joblib.load('student_success_pipeline.pkl')
    model            = bundle['pipeline']
    threshold        = bundle['tuned_threshold']
    expected_columns = bundle['feature_names']
    st.success(f"Model loaded successfully (threshold = {threshold:.2f})")
except Exception as e:
    st.error("Failed to load model")
    st.code(traceback.format_exc())
    st.stop()

st.write(
    "This tool predicts whether a student is likely to **Pass** or be **At-Risk** "
    "based on their learning interaction data. Enter the student's metrics below "
    "and click **Predict**."
)
st.divider()

# =====================================================
# INPUT FORM
# =====================================================
st.subheader("Student Learning Interaction Data")

col1, col2 = st.columns(2)

with col1:
    time_spent      = st.number_input("Time spent (minutes)", min_value=0, max_value=500, value=60)
    pages_visited   = st.number_input("Pages visited", min_value=0, max_value=200, value=10)
    video_pct       = st.slider("Video watched (%)", 0.0, 100.0, 50.0, step=0.1)
    click_events    = st.number_input("Click events", min_value=0, max_value=1000, value=20)
    notes_taken     = st.number_input("Notes taken", min_value=0, max_value=100, value=1)
    forum_posts     = st.number_input("Forum posts", min_value=0, max_value=100, value=2)

with col2:
    revisit_flag    = st.selectbox(
        "Revisited content?", options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    attempts_taken  = st.number_input("Attempts taken", min_value=1, max_value=20, value=1)
    feedback_rating = st.slider("Feedback rating (1–5)", 1, 5, 3)
    days_inactive   = st.number_input("Days since last activity", min_value=0, max_value=365, value=2)
    cumul_quiz      = st.number_input("Cumulative quiz score", min_value=0, max_value=5000, value=500)
    attention       = st.slider("Attention score (0–1)", 0.0, 1.0, 0.5, step=0.01)

st.divider()

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔍 Predict", type="primary", use_container_width=True):
    try:
        # 12 raw features from the form
        raw_values = {
            'time_spent_minutes':       time_spent,
            'pages_visited':            pages_visited,
            'video_watched_percent':    video_pct,
            'click_events':             click_events,
            'notes_taken':              notes_taken,
            'forum_posts':              forum_posts,
            'revisit_flag':             revisit_flag,
            'attempts_taken':           attempts_taken,
            'feedback_rating':          feedback_rating,
            'days_since_last_activity': days_inactive,
            'cumulative_quiz_score':    cumul_quiz,
            'attention_score':          attention
        }

        # The 9 behavioural features that get aggregated in the notebook
        # (see Section 1.7 cell 16 - matches the agg_dict)
        aggregated_features = [
            'time_spent_minutes', 'pages_visited', 'video_watched_percent',
            'click_events', 'forum_posts', 'feedback_rating',
            'days_since_last_activity', 'cumulative_quiz_score', 'attention_score'
        ]

        # Synthesize the per-student aggregate features.
        # With a single session as input, mean=min=max=the value itself,
        # and std=0 (no variance with a single observation).
        synthesized = {}
        for feat in aggregated_features:
            v = raw_values[feat]
            synthesized[f'{feat}_mean_per_student'] = v
            synthesized[f'{feat}_std_per_student']  = 0.0
            synthesized[f'{feat}_min_per_student']  = v
            synthesized[f'{feat}_max_per_student']  = v

        # Synthesize the per-student count totals
        synthesized['total_sessions']    = 1
        synthesized['total_revisits']    = revisit_flag
        synthesized['total_notes']       = notes_taken
        synthesized['total_forum_posts'] = forum_posts

        # Combine raw + synthesized into the full 64-feature input
        all_values = {**raw_values, **synthesized}

        # Force column order from the saved bundle
        input_data = pd.DataFrame([all_values])[expected_columns]

        # Use the tuned threshold
        proba_pass = model.predict_proba(input_data)[0][1]
        prediction = 1 if proba_pass >= threshold else 0

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ **PASS**")
            st.write(
                "The model predicts this student is on track for academic success "
                "based on their learning interaction pattern."
            )
        else:
            st.error("⚠️ **AT-RISK**")
            st.write(
                "The model predicts this student may be at-risk of academic failure. "
                "Early intervention is recommended."
            )

        with st.expander("Model details (advanced)"):
            st.write(f"**Predicted Pass probability:** {proba_pass:.1%}")
            st.write(f"**Predicted At-Risk probability:** {(1 - proba_pass):.1%}")
            st.write(f"**Decision threshold:** {threshold:.2f}")
            st.caption(
                "Note: this single-session interface synthesizes the per-student "
                "aggregate features used during training by treating the entered "
                "session as the student's complete history. Predictions may "
                "differ from those obtained in a multi-session batch context."
            )

    except Exception as e:
        st.error("Prediction failed")
        st.code(traceback.format_exc())

st.divider()
st.caption("FYP Project — Student Success Prediction using Behavioural Learning Data")