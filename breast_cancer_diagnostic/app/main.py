import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


def get_clean_data():
    data=pd.read_csv(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\data\data.csv")
    
    data=data.drop(columns=["Unnamed: 32","id"])

    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})
    
    return data

def add_sidebar():
        st.sidebar.header("Cell Nuclei Measurements")
        data=get_clean_data()
        input_dict={}
        slider_labels = [
    # Mean features
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),

    # SE features
        ("Radius (SE)", "radius_se"),
        ("Texture (SE)", "texture_se"),
        ("Perimeter (SE)", "perimeter_se"),
        ("Area (SE)", "area_se"),
        ("Smoothness (SE)", "smoothness_se"),
        ("Compactness (SE)", "compactness_se"),
        ("Concavity (SE)", "concavity_se"),
        ("Concave points (SE)", "concave points_se"),
        ("Symmetry (SE)", "symmetry_se"),
        ("Fractal dimension (SE)", "fractal_dimension_se"),

    # Worst features
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
        ]
        for label,key in slider_labels:
            input_dict[key]=st.sidebar.slider(
                 label,min_value=float(0),
                 max_value=float(data[key].max()),
                 value=float(data[key].mean())
            )
        return input_dict

def get_scaled_values(input_dict):
    data=get_clean_data()
    x=data.drop(["diagnosis"],axis=1)
    scaled_dict={}
    for key,value in input_dict.items():
        max_val=x[key].max()
        min_val=x[key].min()
        scaled_value=(value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict


def get_radar_chart(input_data):
    
    input_data=get_scaled_values(input_data)

    categories = ['Radius','Texture','Perimeter','Area',
                  'Smoothness','Compactness','Concavity',
                  'Concave Points','Symmetry','Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
          input_data["radius_mean"], input_data["texture_mean"], input_data["perimeter_mean"], input_data["area_mean"],
          input_data["smoothness_mean"], input_data["compactness_mean"], input_data["concavity_mean"], input_data["concave points_mean"],input_data["symmetry_mean"],input_data["fractal_dimension_mean"]
        ],
      theta=categories,
      fill='toself',
      name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[
          input_data["radius_se"], input_data["texture_se"], 
          input_data ["perimeter_se"], input_data["area_se"],
          input_data["smoothness_se"], input_data["compactness_se"], input_data["concavity_se"], input_data["concave points_se"],input_data["symmetry_se"],input_data["fractal_dimension_se"]
      ],
      theta=categories,
      fill='toself',
      name='standard error'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[
          input_data["radius_worst"], input_data["texture_worst"], 
          input_data ["perimeter_worst"], input_data["area_worst"],
          input_data["smoothness_worst"], input_data["compactness_worst"], input_data["concavity_worst"], input_data["concave points_worst"],input_data["symmetry_worst"],input_data["fractal_dimension_worst"]
      ],
      theta=categories,
      fill='toself',
      name='worst value'
    ))
    fig.update_layout(
    polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
    showlegend=True
    )

    return fig

def add_predictions(input_data):

    model=pickle.load(open(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\model\model.pkl","rb"))
    scaler=pickle.load(open(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\model\scaler.pkl","rb"))

    input_array=np.array(list(input_data.values())).reshape(1,-1)
    # st.write(input_array)
    input_array_scaled=scaler.transform(input_array)

    st.subheader("Cell Cluster Prediction")
    st.write("the cell cluster is:")
    prediction=model.predict(input_array_scaled)
    
    if prediction[0] == 0:
        st.markdown("<div class='result benign'>Benign</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result malignant'>Malignant</div>", unsafe_allow_html=True)
    
    st.write("probability of being benign is:",model.predict_proba(input_array_scaled)[0][0])
    
    st.write("probability of being malicious is:",model.predict_proba(input_array_scaled)[0][1])
    
    st.write("this app can assist medical professionals in making a diagnosis,but should not be used as a substitute for a professional diagnosis")

def get_correlation_heatmap():
    data = get_clean_data()
    corr = data.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        labels=dict(color="Correlation"),
        aspect="auto"
    )

    fig.update_layout(
        width=1000,
        height=800,
        xaxis_title="Features",
        yaxis_title="Features",
        margin=dict(l=60, r=60, t=60, b=60)
    )

    return fig

def plot_density_plot():
    st.subheader("Feature Density Comparison")
    data = get_clean_data()
    # Dropdown to select feature
    feature = st.selectbox(
        "Select a feature to view its density distribution:",
        [col for col in data.columns if col not in ["diagnosis"]]
    )

    # Create the density plot
    fig = px.histogram(
        data,
        x=feature,
        color="diagnosis",
        marginal="violin",  # adds a small violin plot above
        opacity=0.6,
        barmode="overlay",
        histnorm="density",
        nbins=40,
        color_discrete_map={0: 'blue', 1: 'red'},
        title=f"Density Distribution of {feature.replace('_',' ').title()} (Benign vs Malignant)"
    )

    fig.update_layout(
        xaxis_title=feature.replace("_", " ").title(),
        yaxis_title="Density",
        bargap=0.1,
        height=600
    )
    return fig

def plot_feature_importance():
    st.subheader("Feature Importance (Model Explainability)")

    # Load the model
    model = pickle.load(open(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\model\model.pkl", "rb"))
    data = get_clean_data()
    feature_names = data.drop("diagnosis", axis=1).columns

    # Get coefficients from logistic regression
    importance = model.coef_[0]

    # Create dataframe
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    # Plot bar chart
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="RdBu_r",
        title="Feature Importance from Logistic Regression Model"
    )

    fig.update_layout(
        xaxis_title="Coefficient Value (Impact on Malignancy)",
        yaxis_title="Feature",
        height=800
    )

    return fig

def main():
    # print("we are going to build streamlit app here")
    st.set_page_config(
        page_title="Breast Cancer Diagnostic App",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
        )
    # st.write("hello world")
   
    with open(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\app\styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    with st.container():
        st.title("Breast Cancer Diagnostic App")
        st.write("An interactive diagnostic tool that analyzes medical features to predict whether a breast tumor is benign or malignant using a Logistic Regression model.")


    col1,col2=st.columns([4,1])

    input_data=add_sidebar()
    # st.write(input_data)

    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
    
    with st.container():
        st.subheader("Feature Correlation Heatmap")
        st.write("This shows how strongly each measurement is correlated with others.")
        corr_chart = get_correlation_heatmap()
        st.plotly_chart(corr_chart, use_container_width=True)
    
    with st.container():
        density_plot = plot_density_plot()
        st.plotly_chart(density_plot, use_container_width=True)

    with st.container():
        feature_importance_plot = plot_feature_importance()
        st.plotly_chart(feature_importance_plot, use_container_width=True)

    with st.container():
        data = get_clean_data()
        st.header("Data Exploration")
        st.write("This section displays the raw dataset and basic statistics.")

        st.subheader("Raw Dataset")
        st.dataframe(data.head())
        st.subheader("Dataset Info")
        st.write(f"*Rows:* {data.shape[0]}  |  *Columns:* {data.shape[1]}")
        st.write("### Missing Values")
        st.write(data.isnull().sum())

        # Diagnosis Count
        st.subheader("Class Distribution")
        fig = px.pie(data, names="diagnosis", title="Diagnosis Distribution")
        st.plotly_chart(fig, use_container_width=True)

        
        st.header("⚙ Data Cleaning & Preprocessing")
    
        st.markdown("""
        ### ✔ Steps Performed:
        - Removed unnecessary columns: *id, **Unnamed: 32*
        - Converted diagnosis column:
            - *M → 1 (Malignant)*
            - *B → 0 (Benign)*
        - Standardized numerical features using *StandardScaler*
        - Split the data into *train/test* using 80/20 ratio
        """)

        cleaned_data = data.copy()
        if "id" in cleaned_data.columns:
            cleaned_data = cleaned_data.drop(columns=["id"])
        if "Unnamed: 32" in cleaned_data.columns:
            cleaned_data = cleaned_data.drop(columns=["Unnamed: 32"])

        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_data.head())

        st.subheader("Scaled Feature Example")
        sample = cleaned_data.drop(columns=["diagnosis"]).head()
        scaler = pickle.load(open(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\model\scaler.pkl","rb"))
        scaled_sample = scaler.transform(sample)
        st.write(pd.DataFrame(scaled_sample, columns=sample.columns).head())

if __name__=="__main__":
    main()