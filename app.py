import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import os

# ==================== BACKEND CONFIGURATION ====================
# MODIFY THESE PATHS - User only uploads CSV
MODEL_PATH = r"D:\ML Micro project\final_rul_model_compatible.h5"  # Change this to your model path
SCALER_PATH = r"D:\ML Micro project\scaler_fixed.pkl"          # Change this to your scaler path
WINDOW_SIZE = 60                          # Must match training window size

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="RUL Prediction System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL AND SCALER ====================
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """Load the trained model and scaler - FIXED for TensorFlow compatibility"""
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        if not os.path.exists(scaler_path):
            return None, None, f"Scaler file not found: {scaler_path}"
        
        # Load model with compile=False to avoid loss function issues
        # FIX: Use custom_objects=None to avoid InputLayer deserialization issues
        model = load_model(model_path, compile=False)
        
        # Recompile with MSE (matching your training code)
        model.compile(
            optimizer='adam',
            loss='mse',  # Changed from asymmetric_loss to match training
            metrics=['mae']
        )
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        return model, scaler, None
        
    except Exception as e:
        return None, None, f"Error loading files: {str(e)}"

# ==================== PREDICTION FUNCTION ====================
def predict_rul(df, model, scaler, window_size=60):
    """
    Make RUL predictions using sliding window approach
    Returns: results_df with all predictions, mean_rul_per_engine, statistics
    """
    # Define sensor columns (exclude engine_id, cycle, and RUL/rul if present)
    sensor_cols = [col for col in df.columns if col not in ["engine_id", "cycle", "RUL", "rul"]]
    
    # Store predictions for all sliding windows
    all_predictions = []
    all_engine_ids = []
    all_start_cycles = []
    all_end_cycles = []
    
    # Error tracking
    error_logs = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    engine_groups = list(df.groupby("engine_id"))
    total_engines = len(engine_groups)
    processed_engines_count = 0
    skipped_engines_count = 0
    failed_windows = 0
    
    for idx, (engine_id, group) in enumerate(engine_groups):
        group = group.sort_values('cycle').reset_index(drop=True)
        n_rows = len(group)
        
        # Update progress
        progress = (idx + 1) / total_engines
        progress_bar.progress(progress)
        status_text.text(f"Processing engine {idx + 1}/{total_engines} (ID: {engine_id}) - {n_rows} cycles")
        
        # Skip engines with insufficient cycles
        if n_rows < window_size:
            skipped_engines_count += 1
            error_logs.append(f"Skipped engine {engine_id}: Only {n_rows} cycles (need {window_size})")
            continue
        
        processed_engines_count += 1
        
        # Apply scaling to the entire engine group's sensor data once
        try:
            group_scaled = group.copy()
            group_scaled[sensor_cols] = scaler.transform(group_scaled[sensor_cols])
        except Exception as e:
            error_msg = f"Error scaling data for engine {engine_id}: {str(e)}"
            error_logs.append(error_msg)
            skipped_engines_count += 1
            continue
        
        # Slide over the group rows with window size
        for start_idx in range(n_rows - window_size + 1):
            try:
                # Get the scaled window
                window_scaled_df = group_scaled.iloc[start_idx:start_idx + window_size]
                
                # Extract features as numpy array
                window_features_scaled = window_scaled_df[sensor_cols].values
                
                # Reshape to (1, window_size, num_features)
                X = window_features_scaled.reshape(1, window_size, len(sensor_cols))
                
                # Predict RUL
                pred_rul = model.predict(X, verbose=0)[0][0]
                
                # Store results
                all_predictions.append(float(pred_rul))
                all_engine_ids.append(int(engine_id))
                all_start_cycles.append(int(window_scaled_df["cycle"].iloc[0]))
                all_end_cycles.append(int(window_scaled_df["cycle"].iloc[-1]))
                
            except Exception as e:
                failed_windows += 1
                error_msg = f"Prediction failed for engine {engine_id}, window starting at cycle {start_idx}: {str(e)}"
                error_logs.append(error_msg)
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Create results dataframe
    if all_engine_ids:
        results_df = pd.DataFrame({
            "engine_id": all_engine_ids,
            "start_cycle": all_start_cycles,
            "end_cycle": all_end_cycles,
            "predicted_rul": all_predictions
        })
        
        # Calculate comprehensive statistics per engine
        mean_rul_per_engine = results_df.groupby("engine_id")["predicted_rul"].agg([
            ('mean_predicted_rul', 'mean'),
            ('min_predicted_rul', 'min'),
            ('max_predicted_rul', 'max'),
            ('std_predicted_rul', 'std'),
            ('num_predictions', 'count')
        ]).reset_index()
        
        # Add health status based on mean RUL
        def categorize_health(rul):
            if rul < 30:
                return "🔴 Critical"
            elif rul < 60:
                return "🟡 Warning"
            else:
                return "🟢 Healthy"
        
        mean_rul_per_engine['health_status'] = mean_rul_per_engine['mean_predicted_rul'].apply(categorize_health)
        
        stats = {
            'processed': processed_engines_count,
            'skipped': skipped_engines_count,
            'failed_windows': failed_windows,
            'total_predictions': len(results_df),
            'error_logs': error_logs
        }
        
        return results_df, mean_rul_per_engine, stats
    else:
        return None, None, {
            'processed': 0,
            'skipped': skipped_engines_count,
            'failed_windows': failed_windows,
            'total_predictions': 0,
            'error_logs': error_logs
        }

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">🔧 Remaining Useful Life (RUL) Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LSTM-based Predictive Maintenance for Turbofan Engines</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ System Status")
    st.sidebar.markdown("---")
    
    # Load model and scaler at startup
    with st.spinner("Loading AI model..."):
        model, scaler, error = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    
    if error:
        st.sidebar.markdown(f'<div class="error-box">❌ <b>System Error</b><br>{error}</div>', unsafe_allow_html=True)
        st.error(f"""
        ### ❌ System Configuration Error
        
        **Error:** {error}
        
        **Action Required:**
        1. Ensure model file exists at: `{MODEL_PATH}`
        2. Ensure scaler file exists at: `{SCALER_PATH}`
        3. Update paths in the code if files are in different location
        
        **Current Working Directory:** `{os.getcwd()}`
        """)
        st.stop()
    else:
        st.sidebar.markdown("""
        <div class="success-box">
            ✅ <b>System Ready</b><br>
            • Model loaded<br>
            • Scaler loaded<br>
            • Ready for predictions
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Configuration")
    st.sidebar.info(f"""
    **Window Size:** {WINDOW_SIZE} cycles  
    **Sensor Features:** Auto-detected  
    **Model Type:** LSTM Neural Network  
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        This system predicts Remaining Useful Life (RUL) 
        of turbofan engines using deep learning.
        
        **Simply:**
        1. Upload your CSV file
        2. Click "Run Predictions"
        3. View and download results
        """
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Results & Analysis", "📖 Help"])
    
    with tab1:
        st.header("Upload Test Data")
        
        st.markdown("""
        <div class="info-box">
            📋 <b>CSV Format Requirements:</b><br>
            • Must contain columns: <code>engine_id</code>, <code>cycle</code><br>
            • Must contain sensor columns (auto-detected)<br>
            • Each engine should have at least <b>60 cycles</b> of data
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your CSV file",
            type=['csv'],
            help="Upload CSV with engine_id, cycle, and sensor data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
                
                # Validate required columns
                if 'engine_id' not in df.columns or 'cycle' not in df.columns:
                    st.markdown("""
                    <div class="error-box">
                        ❌ <b>Invalid CSV Format</b><br>
                        Missing required columns: <code>engine_id</code> and/or <code>cycle</code>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
                # Detect sensor columns
                sensor_cols = [col for col in df.columns if col not in ["engine_id", "cycle", "RUL", "rul"]]
                
                if len(sensor_cols) == 0:
                    st.markdown("""
                    <div class="error-box">
                        ❌ <b>No Sensor Columns Detected</b><br>
                        CSV must contain sensor data columns
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
                # Display data info
                st.subheader("📊 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Engines", df['engine_id'].nunique())
                with col3:
                    st.metric("Sensor Features", len(sensor_cols))
                with col4:
                    cycles_per_engine = df.groupby('engine_id')['cycle'].count()
                    st.metric("Avg Cycles/Engine", f"{cycles_per_engine.mean():.0f}")
                
                # Check for engines with insufficient data
                cycles_per_engine = df.groupby('engine_id').size()
                insufficient_engines = (cycles_per_engine < WINDOW_SIZE).sum()
                
                if insufficient_engines > 0:
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ <b>Warning:</b> {insufficient_engines} engine(s) have fewer than {WINDOW_SIZE} cycles and will be skipped
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show data preview
                with st.expander("📋 Preview Data (First 20 rows)"):
                    st.dataframe(df.head(20), use_container_width=True)
                
                # Show detected columns
                with st.expander("🔍 Detected Columns"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Identifier Columns:**")
                        st.write("• engine_id")
                        st.write("• cycle")
                        if 'RUL' in df.columns or 'rul' in df.columns:
                            st.write("• RUL (will be excluded from features)")
                    with col2:
                        st.markdown(f"**Sensor Columns ({len(sensor_cols)}):**")
                        for i, col in enumerate(sensor_cols[:15]):
                            st.write(f"• {col}")
                        if len(sensor_cols) > 15:
                            st.write(f"... and {len(sensor_cols) - 15} more")
                
                st.markdown("---")
                
                # Predict button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                        with st.spinner("🔄 Making predictions... This may take a few minutes..."):
                            results_df, mean_rul_per_engine, stats = predict_rul(
                                df, model, scaler, WINDOW_SIZE
                            )
                        
                        if results_df is not None:
                            # Store in session state
                            st.session_state['results_df'] = results_df
                            st.session_state['mean_rul_per_engine'] = mean_rul_per_engine
                            st.session_state['original_df'] = df
                            st.session_state['stats'] = stats
                            
                            # Success message
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ <b>Predictions Completed!</b><br><br>
                                📊 <b>Summary:</b><br>
                                • Engines processed: <b>{stats['processed']}</b><br>
                                • Engines skipped: <b>{stats['skipped']}</b><br>
                                • Total predictions: <b>{stats['total_predictions']:,}</b><br>
                                • Failed windows: <b>{stats['failed_windows']}</b>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show critical engines immediately
                            critical = mean_rul_per_engine[mean_rul_per_engine['health_status'] == '🔴 Critical']
                            if len(critical) > 0:
                                st.markdown(f"""
                                <div class="warning-box">
                                    🚨 <b>ALERT: {len(critical)} engine(s) require immediate attention!</b><br>
                                    Critical engines: {', '.join(map(str, critical['engine_id'].tolist()))}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show error log if needed
                            if stats['error_logs']:
                                with st.expander(f"⚠️ View Processing Log ({len(stats['error_logs'])} messages)"):
                                    for log in stats['error_logs'][:100]:
                                        st.text(log)
                                    if len(stats['error_logs']) > 100:
                                        st.info(f"... and {len(stats['error_logs']) - 100} more messages")
                            
                            st.success("✅ Go to the 'Results & Analysis' tab to explore predictions!")
                            
                        else:
                            st.markdown("""
                            <div class="error-box">
                                ❌ <b>No Predictions Generated</b><br><br>
                                Possible reasons:<br>
                                • All engines have fewer than 60 cycles<br>
                                • Sensor data format mismatch<br>
                                • Check the error log above for details
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ <b>Error Loading File:</b><br>{str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📊 Prediction Results & Analysis")
        
        if 'results_df' not in st.session_state:
            st.info("👆 Please upload data and run predictions in the 'Upload & Predict' tab first.")
            st.stop()
        
        results_df = st.session_state['results_df']
        mean_rul_per_engine = st.session_state['mean_rul_per_engine']
        stats = st.session_state['stats']
        
        # Overall Summary
        st.subheader("📈 Overall Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Predictions", f"{len(results_df):,}")
        with col2:
            st.metric("Engines Analyzed", results_df['engine_id'].nunique())
        with col3:
            st.metric("Mean RUL", f"{results_df['predicted_rul'].mean():.1f}")
        with col4:
            critical_count = len(mean_rul_per_engine[mean_rul_per_engine['health_status'] == '🔴 Critical'])
            st.metric("Critical Engines", critical_count)
        with col5:
            healthy_count = len(mean_rul_per_engine[mean_rul_per_engine['health_status'] == '🟢 Healthy'])
            st.metric("Healthy Engines", healthy_count)
        
        st.markdown("---")
        
        # Health Status Distribution
        st.subheader("🏥 Fleet Health Status")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            health_counts = mean_rul_per_engine['health_status'].value_counts()
            st.dataframe(
                health_counts.rename("Count").to_frame(),
                use_container_width=True
            )
        
        with col2:
            fig = px.pie(
                values=health_counts.values,
                names=health_counts.index,
                title='Fleet Health Distribution',
                color_discrete_map={
                    '🔴 Critical': '#dc3545',
                    '🟡 Warning': '#ffc107',
                    '🟢 Healthy': '#28a745'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Engine-level Results
        st.subheader("🔍 Detailed Engine Results")
        
        # Sort options
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Mean RUL (Low to High)", "Mean RUL (High to Low)", "Engine ID"]
            )
        
        # Apply sorting
        if sort_by == "Mean RUL (Low to High)":
            display_df = mean_rul_per_engine.sort_values('mean_predicted_rul', ascending=True)
        elif sort_by == "Mean RUL (High to Low)":
            display_df = mean_rul_per_engine.sort_values('mean_predicted_rul', ascending=False)
        else:
            display_df = mean_rul_per_engine.sort_values('engine_id')
        
        # FIXED: Display table without background_gradient (no matplotlib required)
        # Option 1: Simple formatting without gradient
        st.dataframe(
            display_df.style.format({
                'mean_predicted_rul': '{:.2f}',
                'min_predicted_rul': '{:.2f}',
                'max_predicted_rul': '{:.2f}',
                'std_predicted_rul': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Visual Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution histogram
            fig1 = px.histogram(
                results_df,
                x='predicted_rul',
                nbins=50,
                title='Distribution of All RUL Predictions',
                labels={'predicted_rul': 'Predicted RUL (cycles)', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Box plot
            fig2 = px.box(
                mean_rul_per_engine,
                y='mean_predicted_rul',
                title='Mean RUL Distribution Across Engines',
                labels={'mean_predicted_rul': 'Mean Predicted RUL (cycles)'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Top/Bottom engines
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Top 10 Critical Engines")
            lowest_rul = mean_rul_per_engine.nsmallest(10, 'mean_predicted_rul')
            fig3 = px.bar(
                lowest_rul,
                x='engine_id',
                y='mean_predicted_rul',
                color='mean_predicted_rul',
                color_continuous_scale='Reds_r',
                labels={'engine_id': 'Engine ID', 'mean_predicted_rul': 'Mean RUL'}
            )
            fig3.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.markdown("#### 🟢 Top 10 Healthiest Engines")
            highest_rul = mean_rul_per_engine.nlargest(10, 'mean_predicted_rul')
            fig4 = px.bar(
                highest_rul,
                x='engine_id',
                y='mean_predicted_rul',
                color='mean_predicted_rul',
                color_continuous_scale='Greens',
                labels={'engine_id': 'Engine ID', 'mean_predicted_rul': 'Mean RUL'}
            )
            fig4.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("---")
        
        # Individual engine deep dive
        st.subheader("🔎 Individual Engine Analysis")
        
        selected_engine = st.selectbox(
            "Select an engine to analyze:",
            options=sorted(results_df['engine_id'].unique()),
            format_func=lambda x: f"Engine {x}"
        )
        
        if selected_engine:
            engine_data = results_df[results_df['engine_id'] == selected_engine].copy()
            engine_summary = mean_rul_per_engine[mean_rul_per_engine['engine_id'] == selected_engine].iloc[0]
            
            # Engine metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Mean RUL", f"{engine_summary['mean_predicted_rul']:.1f}")
            with col2:
                st.metric("Min RUL", f"{engine_summary['min_predicted_rul']:.1f}")
            with col3:
                st.metric("Max RUL", f"{engine_summary['max_predicted_rul']:.1f}")
            with col4:
                st.metric("Std Dev", f"{engine_summary['std_predicted_rul']:.1f}")
            with col5:
                st.metric("Windows", int(engine_summary['num_predictions']))
            
            # Health status badge
            st.markdown(f"**Health Status:** {engine_summary['health_status']}")
            
            # RUL over time plot
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=engine_data['start_cycle'],
                y=engine_data['predicted_rul'],
                mode='lines+markers',
                name='Predicted RUL',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            # Add trend line
            z = np.polyfit(engine_data['start_cycle'], engine_data['predicted_rul'], 1)
            p = np.poly1d(z)
            fig5.add_trace(go.Scatter(
                x=engine_data['start_cycle'],
                y=p(engine_data['start_cycle']),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig5.update_layout(
                title=f'RUL Predictions Over Time - Engine {selected_engine}',
                xaxis_title='Start Cycle',
                yaxis_title='Predicted RUL (cycles)',
                hovermode='x unified',
                height=450
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            # Detailed predictions table
            with st.expander("📋 View All Predictions for This Engine"):
                st.dataframe(
                    engine_data.style.format({'predicted_rul': '{:.2f}'}),
                    use_container_width=True,
                    height=400
                )
        
        st.markdown("---")
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv1 = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 All Predictions",
                data=csv1,
                file_name="rul_all_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download all sliding window predictions"
            )
        
        with col2:
            csv2 = mean_rul_per_engine.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Engine Summary",
                data=csv2,
                file_name="rul_engine_summary.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download aggregated results per engine"
            )
        
        with col3:
            # Critical engines only
            critical_engines = mean_rul_per_engine[mean_rul_per_engine['health_status'] == '🔴 Critical']
            if len(critical_engines) > 0:
                csv3 = critical_engines.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Critical Engines",
                    data=csv3,
                    file_name="rul_critical_engines.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download only critical engines"
                )
            else:
                st.button(
                    label="📥 Critical Engines",
                    disabled=True,
                    use_container_width=True,
                    help="No critical engines found"
                )
    
    with tab3:
        st.header("📖 Help & Documentation")
        
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        ### Step 1: Prepare Your CSV File
        
        Your CSV file must contain:
        - **`engine_id`**: Unique identifier for each engine (integer)
        - **`cycle`**: Time step or cycle number (integer)
        - **Sensor columns**: Multiple columns with sensor readings (numeric)
        
        **Example CSV Structure:**
        ```
        engine_id,cycle,sensor_1,sensor_2,sensor_3,...
        1,1,0.5,0.3,0.8,...
        1,2,0.6,0.4,0.7,...
        1,3,0.7,0.5,0.6,...
        2,1,0.4,0.5,0.9,...
        ```
        
        ---
        
        ### Step 2: Upload and Predict
        
        1. Go to the **"Upload & Predict"** tab
        2. Click **"Choose your CSV file"** and select your data
        3. Review the dataset overview and detected columns
        4. Click **"🚀 Run Predictions"**
        5. Wait for processing to complete (progress bar will show status)
        
        ---
        
        ### Step 3: Analyze Results
        
        1. Go to the **"Results & Analysis"** tab
        2. View overall fleet health summary
        3. Examine individual engine predictions
        4. Download results as CSV files
        
        ---
        
        ## 📊 Understanding the Results
        
        ### Health Status Categories
        
        - **🔴 Critical** (RUL < 30 cycles): Immediate maintenance required
        - **🟡 Warning** (30 ≤ RUL < 60 cycles): Schedule maintenance soon
        - **🟢 Healthy** (RUL ≥ 60 cycles): No immediate action needed
        
        ### Output Files
        
        **1. All Predictions CSV**
        - Contains every sliding window prediction
        - Columns: `engine_id`, `start_cycle`, `end_cycle`, `predicted_rul`
        - Use this for detailed temporal analysis
        
        **2. Engine Summary CSV**
        - Aggregated statistics per engine
        - Columns:
          - `engine_id`: Engine identifier
          - `mean_predicted_rul`: Average RUL across all windows
          - `min_predicted_rul`: Minimum predicted RUL
          - `max_predicted_rul`: Maximum predicted RUL
          - `std_predicted_rul`: Standard deviation of predictions
          - `num_predictions`: Number of windows analyzed
          - `health_status`: Health category
        
        **3. Critical Engines CSV**
        - Contains only engines requiring immediate attention
        - Prioritize these for maintenance scheduling
        
        ---
        
        ## 🔧 Technical Details
        
        ### Prediction Method
        
        The system uses a **sliding window approach**:
        
        1. For each engine, the model analyzes windows of **60 consecutive cycles**
        2. Each window produces one RUL prediction
        3. For an engine with 100 cycles:
           - Window 1: cycles 1-60 → predicts RUL
           - Window 2: cycles 2-61 → predicts RUL
           - Window 41: cycles 41-100 → predicts RUL
           - Total: 41 predictions
        4. Final RUL = mean of all window predictions
        
        ### Model Architecture
        
        - **Type**: Long Short-Term Memory (LSTM) Neural Network
        - **Input**: 60 cycles × N sensor features
        - **Output**: Remaining useful life (cycles)
        - **Loss Function**: Mean Squared Error (MSE)
        - **Optimizer**: Adam
        
        ### Data Processing
        
        1. **Scaling**: All sensor data is normalized using StandardScaler
        2. **Windowing**: Creates overlapping sequences of 60 cycles
        3. **Prediction**: Each window is fed through the LSTM model
        4. **Aggregation**: Results are averaged per engine
        
        ---
        
        ## ⚠️ Troubleshooting
        
        ### "System Configuration Error"
        
        **Problem**: Model or scaler file not found
        
        **Solution**:
        - Ensure `final_rul_model_compatible.h5` exists at: `{MODEL_PATH}`
        - Ensure `scaler_fixed.pkl` exists at: `{SCALER_PATH}`
        - Check file names match exactly (case-sensitive)
        - Update `MODEL_PATH` and `SCALER_PATH` at the top of the code if files are elsewhere
        
        ---
        
        ### "Invalid CSV Format"
        
        **Problem**: Missing required columns
        
        **Solution**:
        - CSV must have `engine_id` column (exact spelling, case-sensitive)
        - CSV must have `cycle` column (exact spelling, case-sensitive)
        - Check for typos in column names
        
        ---
        
        ### "No Predictions Generated"
        
        **Problem**: All engines were skipped
        
        **Possible Causes**:
        1. **Insufficient data**: Each engine needs at least 60 cycles
        2. **Sensor column mismatch**: Sensor features don't match training data
        3. **Data format issues**: Non-numeric values in sensor columns
        
        **Solution**:
        - Check "Processing Log" for specific errors
        - Ensure each engine has ≥60 rows in the CSV
        - Verify sensor columns contain only numeric data
        - Remove or fill missing values (NaN)
        
        ---
        
        ### High Number of "Failed Windows"
        
        **Problem**: Some predictions failed during processing
        
        **Possible Causes**:
        - Missing values (NaN) in sensor data
        - Infinite values (inf) in sensor data
        - Sensor column count mismatch with training
        
        **Solution**:
        - Check for NaN values: `df.isnull().sum()`
        - Check for infinite values: `df[sensor_cols].isin([np.inf, -np.inf]).sum()`
        - Ensure sensor column count matches training (auto-detected)
        - Fill or remove problematic rows
        
        ---
        
        ### Predictions Seem Inaccurate
        
        **Possible Causes**:
        1. **Different data distribution**: Test data differs significantly from training data
        2. **Sensor drift**: Sensor calibration or characteristics have changed
        3. **Model limitations**: Model trained on specific engine types/conditions
        
        **What to check**:
        - Compare test data sensor ranges with training data ranges
        - Verify test engines are similar to training engines
        - Check for data preprocessing differences
        - Review model training metrics (RMSE, MAE, R²)
        
        ---
        
        ## 💡 Best Practices
        
        ### Data Quality
        
        ✅ **DO:**
        - Ensure consistent sensor column names
        - Remove or impute missing values before upload
        - Verify data types (numeric for sensors)
        - Include sufficient cycle history (≥60 cycles per engine)
        
        ❌ **DON'T:**
        - Mix different engine types in one file (unless trained together)
        - Upload data with different sensor sets than training
        - Include test engines with <60 cycles
        
        ### Interpreting Results
        
        ✅ **DO:**
        - Focus on the **mean RUL** for maintenance planning
        - Use **health status** for prioritization
        - Monitor **trend** in RUL over time plots
        - Consider **std deviation** (high = uncertain predictions)
        
        ❌ **DON'T:**
        - Make critical decisions on single predictions
        - Ignore engines with high std deviation
        - Assume predictions are 100% accurate (use as guidance)
        
        ### Maintenance Planning
        
        1. **Immediate Action** (🔴 Critical, RUL < 30):
           - Schedule urgent maintenance
           - Increase monitoring frequency
           - Consider operational restrictions
        
        2. **Plan Ahead** (🟡 Warning, 30 ≤ RUL < 60):
           - Schedule maintenance in next window
           - Order spare parts
           - Prepare maintenance crew
        
        3. **Monitor** (🟢 Healthy, RUL ≥ 60):
           - Continue regular operations
           - Maintain scheduled monitoring
           - Re-evaluate in next prediction cycle
        
        ---
        
        ## 📞 System Information
        
        **Model Configuration:**
        - Window Size: {WINDOW_SIZE} cycles
        - Model File: `{MODEL_PATH}`
        - Scaler File: `{SCALER_PATH}`
        
        **Performance Notes:**
        - Processing time: ~1-3 seconds per engine
        - Memory usage: Scales with dataset size
        - Recommended: Upload <100k rows for optimal performance
        
        ---
        
        ## 📚 Additional Resources
        
        ### Understanding RUL Prediction
        
        Remaining Useful Life (RUL) prediction is a key component of predictive maintenance:
        
        - **Goal**: Estimate how many cycles until engine failure/maintenance needed
        - **Benefits**: Reduce downtime, optimize maintenance schedules, prevent failures
        - **Approach**: Machine learning on historical sensor data and failure patterns
        
        ### LSTM Networks
        
        Long Short-Term Memory (LSTM) networks are ideal for RUL prediction because:
        - Can learn long-term dependencies in sensor data
        - Captures degradation patterns over time
        - Handles sequential/time-series data naturally
        
        ### Model Training
        
        This model was trained on the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset:
        - Turbofan engine degradation data
        - Multiple sensor readings per cycle
        - Run-to-failure trajectories
        - Standard benchmark for predictive maintenance research
        
        ---
        
        ## 🔄 Updates & Maintenance
        
        To update the model with new data:
        1. Retrain the model using the training script
        2. Replace `{MODEL_PATH}` with new model file
        3. Replace `{SCALER_PATH}` with new scaler file
        4. Update `WINDOW_SIZE` if changed
        5. Restart the Streamlit app
        
        ---
        
        ## ⚙️ Backend Configuration
        
        To modify system settings, edit these variables at the top of the code:
        
        ```python
        MODEL_PATH = "final_rul_model_compatible.h5"  # Your model file path
        SCALER_PATH = "scaler_fixed.pkl"              # Your scaler file path
        WINDOW_SIZE = 60                              # Must match training
        ```
        
        After changes, restart the app:
        ```bash
        streamlit run app.py
        ```
        """)

if __name__ == "__main__":
    main()