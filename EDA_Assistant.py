import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind, chi2_contingency
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
# Enable experimental IterativeImputer before importing it
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from transformers import pipeline

# -----------------------------------------------------------------------------
# Initialize NLP Summarizer
# -----------------------------------------------------------------------------
summarizer = pipeline("summarization", model="t5-small")

def generate_nlp_explanation(context: str) -> str:
    """Generate a concise NLP-based explanation/summary for a given context."""
    try:
        words = context.split()
        input_length = len(words)
        max_length = min(250, int(input_length * 1.2))
        if input_length < 30:
            max_length = min(50, input_length + 10)
        summary = summarizer(
            context,
            max_length=max_length,
            min_length=10,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"NLP explanation failed: {e}"

# -----------------------------------------------------------------------------
# Economic Shock Function
# -----------------------------------------------------------------------------
def apply_economic_shock(df: pd.DataFrame, shock_variable: str, shock_value: float) -> pd.DataFrame:
    """Return a copy of df with shock_value added to shock_variable."""
    df_copy = df.copy()
    df_copy[shock_variable] = df_copy[shock_variable] + shock_value
    return df_copy

# -----------------------------------------------------------------------------
# Plotting Helper
# -----------------------------------------------------------------------------
def plot_graphs(df: pd.DataFrame, column: str):
    """Render histogram, boxplot, scatter, line—and pie when appropriate—with NLP insight."""
    st.subheader(f"Visualizations for `{column}`")
    # Histogram
    fig1 = px.histogram(df, x=column, nbins=50, title=f"Histogram of {column}")
    st.plotly_chart(fig1)
    st.write(generate_nlp_explanation(f"The histogram of {column} shows its distribution and potential outliers."))

    # Boxplot
    fig2 = px.box(df, y=column, title=f"Boxplot of {column}")
    st.plotly_chart(fig2)
    st.write(generate_nlp_explanation(f"The boxplot of {column} highlights median, quartiles, and outliers."))

    # Scatter vs first column
    first_col = df.columns[0]
    if first_col != column:
        fig3 = px.scatter(df, x=column, y=first_col, title=f"Scatter: {column} vs {first_col}")
        st.plotly_chart(fig3)
        st.write(generate_nlp_explanation(f"Scatterplot between {column} and {first_col} to identify correlation."))

    # Line chart (requires index as time)
    try:
        fig4 = px.line(df, x=df.index, y=column, title=f"Line Chart of {column}")
        st.plotly_chart(fig4)
        st.write(generate_nlp_explanation(f"Line chart of {column} over its index to analyze trends."))
    except Exception:
        pass  # If index isn’t datetime-like, skip gracefully

    # Pie chart only if low cardinality
    if df[column].nunique() <= 10:
        fig5 = px.pie(df, names=column, title=f"Pie Chart of {column}")
        st.plotly_chart(fig5)
        st.write(generate_nlp_explanation(f"Pie chart of {column} for categorical proportion comparison."))

# -----------------------------------------------------------------------------
# Streamlit App Layout
# -----------------------------------------------------------------------------
st.title("EDA Assistant: AI-Driven Econometric Dashboard")
st.markdown("Unlock insights with data visualization, econometric modeling, and real-time explanations.")

# Sidebar: File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV, XLSX, JSON, or DTA", type=["csv", "xlsx", "json", "dta"])

if uploaded_file:
    # Load dataset based on extension
    ext = uploaded_file.name.split(".")[-1].lower()
    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif ext == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif ext == "json":
            df = pd.read_json(uploaded_file)
        elif ext == "dta":
            df = pd.read_stata(uploaded_file, convert_categoricals=False)
        else:
            st.error("Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Visualization
    vis_col = st.selectbox("Select Column for Visualizations", df.columns)
    plot_graphs(df, vis_col)

    # Sidebar: Regression Controls
    st.sidebar.subheader("📉 Regression Analysis")
    regression_options = [
        "Linear Regression",
        "Multiple Linear Regression",
        "Logistic Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet Regression",
        "Principal Components Regression",
        "Support Vector Regression",
        "Decision Tree Regression",
        "Random Forest Regression",
        "Vector Auto Regression (VAR)",
        "ARIMA Forecasting"
    ]
    regression_type = st.sidebar.selectbox("Regression Type", regression_options)
    dependent_var = st.sidebar.selectbox("Dependent Variable", df.columns)
    independent_vars = st.sidebar.multiselect(
        "Independent Variables",
        [col for col in df.columns if col != dependent_var]
    )

    model = None
    X = Y = None

    if st.sidebar.button("Run Regression"):
        if not independent_vars:
            st.error("Please select at least one independent variable.")
        else:
            # Prepare data
            X = sm.add_constant(df[independent_vars])
            Y = df[dependent_var]

            # Fit model per choice
            try:
                if regression_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000).fit(X, Y)
                elif regression_type in ["Linear Regression", "Multiple Linear Regression"]:
                    model = sm.OLS(Y, X).fit()
                elif regression_type == "Ridge Regression":
                    model = Ridge().fit(X, Y)
                elif regression_type == "Lasso Regression":
                    model = Lasso().fit(X, Y)
                elif regression_type == "ElasticNet Regression":
                    model = ElasticNet().fit(X, Y)
                elif regression_type == "Principal Components Regression":
                    pca = PCA(n_components=min(len(independent_vars), df.shape[0]))
                    X_pca = pca.fit_transform(X)
                    model = sm.OLS(Y, sm.add_constant(X_pca)).fit()
                elif regression_type == "Support Vector Regression":
                    model = SVR().fit(X, Y)
                elif regression_type == "Decision Tree Regression":
                    model = DecisionTreeRegressor().fit(X, Y)
                elif regression_type == "Random Forest Regression":
                    model = RandomForestRegressor().fit(X, Y)
                elif regression_type == "Vector Auto Regression (VAR)":
                    model = VAR(df[independent_vars]).fit()
                elif regression_type == "ARIMA Forecasting":
                    model = ARIMA(Y, order=(1,1,1)).fit()
                else:
                    model = sm.OLS(Y, X).fit()

            except Exception as e:
                st.error(f"Model fitting failed: {e}")
                model = None

            # Display results
            if model is not None:
                st.write("### Regression Results")

                # statsmodels summary
                if hasattr(model, "summary"):
                    try:
                        st.text(model.summary())
                    except Exception as e:
                        st.error(f"Could not render summary: {e}")
                else:
                    # sklearn-style output
                    coefs = np.r_[model.intercept_, model.coef_.ravel()]
                    var_names = ["Intercept"] + independent_vars
                    coef_df = pd.DataFrame({"Variable": var_names, "Coefficient": coefs})
                    st.table(coef_df)
                    if hasattr(model, "score"):
                        st.write(f"R² Score: {model.score(X, Y):.4f}")

                # NLP-based interpretation
                st.write(generate_nlp_explanation(
                    f"Analysis of {regression_type} predicting {dependent_var} from {independent_vars}."
                ))

    # Sidebar: Diagnostic Tests
    st.sidebar.subheader("📊 Diagnostic Tests")

    # Breusch-Pagan
    if st.sidebar.button("Run Breusch-Pagan Test"):
        if model is not None and hasattr(model, "resid") and X is not None:
            lm_stat, lm_pvalue, _, _ = sm.stats.diagnostic.het_breuschpagan(model.resid, X)
            st.write(f"**Breusch-Pagan** — LM: {lm_stat:.3f}, p-value: {lm_pvalue:.3f}")
            st.write(generate_nlp_explanation(
                f"Breusch-Pagan test result: LM={lm_stat:.3f}, p={lm_pvalue:.3f}."
            ))
        else:
            st.error("Requires a statsmodels OLS model with residuals.")

    # VIF
    if st.sidebar.button("Run VIF Test"):
        if independent_vars:
            vif_df = pd.DataFrame({
                "Feature": independent_vars,
                "VIF": [
                    variance_inflation_factor(df[independent_vars].values, i)
                    for i in range(len(independent_vars))
                ]
            })
            st.write("**Variance Inflation Factors**")
            st.table(vif_df)
            st.write(generate_nlp_explanation(f"VIF results: {vif_df.to_dict('records')}"))
        else:
            st.error("Please select independent variables first.")

    # Sidebar: Economic Policy Shock
    st.sidebar.subheader("🚀 Economic Policy Shock")
    shock_var = st.sidebar.selectbox("Variable to Shock", df.columns, index=0)
    shock_val = st.sidebar.number_input("Shock Amount", value=0.0, format="%.4f")
    if st.sidebar.button("Apply Shock"):
        try:
            shocked_df = apply_economic_shock(df, shock_var, shock_val)
            st.write(f"### Shock Applied: `{shock_var}` + {shock_val}")
            st.dataframe(shocked_df.head())
            st.write(generate_nlp_explanation(
                f"Applied an economic shock of +{shock_val} to {shock_var}."
            ))
        except Exception as e:
            st.error(f"Failed to apply shock: {e}")
st.markdown("<div style='text-align: center; color:gray; font-style: italic;'>"
            "<strong>Designed by Jit</strong> | Powered by Econometrics and AI"
            "</div>", unsafe_allow_html=True)
