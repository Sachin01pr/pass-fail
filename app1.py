import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="📘",
    layout="centered"
)

# Title with style
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Result Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Predict PASS or FAIL using Machine Learning</p>",
    unsafe_allow_html=True
)

st.divider()

# Dataset
X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y = ['fail','fail','fail','fail','pass','pass','pass','pass','pass','pass']

# Encode target
label = LabelEncoder()
y_encoded = label.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# UI Input
hours = st.slider("📚 Select study hours:", 0, 12, 5)

# Predict button
if st.button("🔍 Predict Result"):
    result = model.predict([[hours]])[0]
    final_result = label.inverse_transform([result])[0]

    if final_result == 'pass':
        st.success("🎉 PASS! You studied well. Keep it up!")
    else:
        st.error("😐 FAIL! Study harder and try again.")

st.divider()

# Footer
st.markdown(
    "<p style='text-align: center;'>Made with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
