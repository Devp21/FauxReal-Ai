import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from SyntheticDataGenerator import SyntheticDataGenerator  # Import the main class you've built.
from EvaluationMetrics import ModelEvaluator

# ==================== Streamlit App ====================
st.title("Synthetic Data Generation with GANs")
st.write(
    "Upload a CSV dataset, select the target column, and specify the number of samples to "
    "generate synthetic data using CTGAN, VAE, and WGAN."
)

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(data.head())

    # Step 2: Target Column Selection
    target_column = st.selectbox("Select the target column", [None] + list(data.columns))

    # Step 3: Number of Samples Input
    num_samples = st.number_input(
        "Number of synthetic samples to generate",
        min_value=1,
        value=1000,
        step=1,
    )

    # Initialize session state for synthetic data if not already done
    if 'synthetic_data_generated' not in st.session_state:
        st.session_state.synthetic_data_generated = False

    if 'ctgan_data' not in st.session_state:
        st.session_state.ctgan_data = None
    if 'vae_data' not in st.session_state:
        st.session_state.vae_data = None
    if 'wgan_data' not in st.session_state:
        st.session_state.wgan_data = None

    if st.button("Generate Synthetic Data"):
        with st.spinner("Training GANs and generating synthetic data..."):
            generator = SyntheticDataGenerator()
            generator.train_and_generate(
                dataset=data, 
                num_samples=num_samples, 
                output_prefix="synthetic", 
                target_column=target_column
            )

        st.success("Synthetic data generated successfully!")
        st.session_state.synthetic_data_generated = True  # Set the flag to True

        # Load Generated Datasets into session state
        st.session_state.ctgan_data = pd.read_csv("synthetic_ctgan.csv")
        st.session_state.vae_data = pd.read_csv("synthetic_vae.csv")
        st.session_state.wgan_data = pd.read_csv("synthetic_wgan.csv")

    # Step 5: Comparison Visualization
    if st.session_state.synthetic_data_generated:
        st.write("### Real vs. Synthetic Data Distribution")
        selected_feature = st.selectbox(
            "Select a feature for distribution comparison",
            [col for col in data.columns if col != target_column],
        )

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(data[selected_feature], ax=ax[0], kde=True, color="blue", label="Real Data")
        if st.session_state.ctgan_data is not None:
            sns.histplot(st.session_state.ctgan_data[selected_feature], ax=ax[1], kde=True, color="green", label="CTGAN")
        if st.session_state.wgan_data is not None:
            sns.histplot(st.session_state.wgan_data[selected_feature], ax=ax[2], kde=True, color="orange", label="WGAN")
        ax[0].set_title("Real Data")
        ax[1].set_title("CTGAN")
        ax[2].set_title("WGAN")
        plt.tight_layout()
        st.pyplot(fig)

        # Step 6: Download Options
        st.write("### Download Synthetic Data")
        def convert_df_to_csv(df):
            return BytesIO(df.to_csv(index=False).encode("utf-8"))

        st.download_button(
            label="Download CTGAN Synthetic Data",
            data=convert_df_to_csv(st.session_state.ctgan_data),
            file_name="synthetic_ctgan.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download VAE Synthetic Data",
            data=convert_df_to_csv(st.session_state.vae_data),
            file_name="synthetic_vae.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download WGAN Synthetic Data",
            data=convert_df_to_csv(st.session_state.wgan_data),
            file_name="synthetic_wgan.csv",
            mime="text/csv",
        )
        
        st.header("Evaluate Models on Real and Synthetic Data")


        # Check if synthetic data has been generated before allowing evaluation
        if st.session_state.synthetic_data_generated:
            target_column_eval = st.selectbox("Select the target column for evaluation", data.columns)

             # Ensure target column is present in both real and synthetic datasets
            if target_column_eval not in data.columns:
                st.error(f"Target column '{target_column_eval}' not found in real data.")

                # Ensure target column exists in synthetic datasets
            for name, dataset in {
                "CTGAN": st.session_state.ctgan_data,
                "WGAN": st.session_state.wgan_data,
                "VAE": st.session_state.vae_data
            }.items():
                if target_column_eval not in dataset.columns:
                     st.error(f"Target column '{target_column_eval}' not found in synthetic dataset '{name}'.")

            
            # Proceed with evaluation when button is clicked
            if st.button("Run Evaluation"):
                # Initialize evaluators for each synthetic dataset
                evaluator_ctgan = ModelEvaluator(data, {"CTGAN": st.session_state.ctgan_data}, target_column_eval)
                evaluator_wgan = ModelEvaluator(data, {"WGAN": st.session_state.wgan_data}, target_column_eval)
                evaluator_vae = ModelEvaluator(data, {"VAE": st.session_state.vae_data}, target_column_eval)

                # Run evaluations and get results
                ctgan_results = evaluator_ctgan.run_evaluation()
                wgan_results = evaluator_wgan.run_evaluation()
                vae_results = evaluator_vae.run_evaluation()

                 # Display the evaluation results
                st.write("### Evaluation Results")
                st.write("#### CTGAN Results")
                st.write(ctgan_results)

                st.write("#### WGAN Results")
                st.write(wgan_results)

                st.write("#### VAE Results")
                st.write(vae_results)


        

