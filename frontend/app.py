
"""
Enhanced Streamlit frontend for Artifact ATP
Provides interactive interface for reasoning, visualization, and proof exploration
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="Artifact Reason - AI Reasoner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'reasoning_results' not in st.session_state:
    st.session_state.reasoning_results = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Artifact Reason — Interactive AI Reasoner</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Reasoning System with Multi-Agent Validation & Formal Proof Verification**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("API Settings")
        api_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
        st.session_state.api_base_url = api_url
        
        # Check API health
        health_status = check_api_health(api_url)
        if health_status["status"] == "healthy":
            st.success(f"✅ API Connected (v{health_status.get('version', 'unknown')})")
        else:
            st.error("❌ API Unavailable")
        
        # Analysis Configuration
        st.subheader("Analysis Settings")
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["basic", "standard", "deep"],
            index=1,
            help="Basic: Traditional pipeline only\nStandard: Multi-agent validation\nDeep: Full analysis with extended proof search"
        )
        
        enable_multi_agent = st.checkbox(
            "Enable Multi-Agent Reasoning",
            value=True,
            help="Use multiple AI agents for validation and consensus building"
        )
        
        timeout_seconds = st.slider(
            "Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            help="Maximum time to spend on analysis"
        )
        
        # Data Input Options
        st.subheader("Data Input")
        input_method = st.radio(
            "Input Method",
            ["Manual Entry", "CSV Upload", "Example Datasets"],
            help="Choose how to provide your data"
        )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Visualization", "📜 Proof Explorer", "📈 History"])
    
    with tab1:
        run_analysis_tab(input_method, analysis_depth, enable_multi_agent, timeout_seconds)
    
    with tab2:
        run_visualization_tab()
    
    with tab3:
        run_proof_explorer_tab()
    
    with tab4:
        run_history_tab()

def check_api_health(api_url: str) -> Dict[str, Any]:
    """Check API health status"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def run_analysis_tab(input_method: str, analysis_depth: str, enable_multi_agent: bool, timeout_seconds: int):
    """Main analysis tab"""
    
    st.header("🔍 Data Analysis & Reasoning")
    
    # Data input section
    data = get_data_input(input_method)
    
    if data is not None and len(data) > 0:
        # Display data preview
        st.subheader("📋 Data Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show data as DataFrame
            df = pd.DataFrame({
                'Index': range(len(data)),
                'Value': data,
                'Difference': [None] + [data[i] - data[i-1] for i in range(1, len(data))]
            })
            st.dataframe(df, use_container_width=True)
        
        with col2:
            # Basic statistics
            st.metric("Data Points", len(data))
            st.metric("Min Value", f"{min(data):.3f}")
            st.metric("Max Value", f"{max(data):.3f}")
            st.metric("Mean", f"{np.mean(data):.3f}")
            st.metric("Std Dev", f"{np.std(data):.3f}")
        
        # Quick data visualization
        fig = px.line(x=range(len(data)), y=data, title="Data Visualization")
        fig.update_traces(mode='lines+markers')
        fig.update_layout(xaxis_title="Index", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Known theorems input
        st.subheader("📚 Known Theorems (Optional)")
        theorems_input = st.text_area(
            "Enter known theorems (JSON format)",
            placeholder='[{"name": "Linear", "structure": "y = mx + b"}]',
            help="Provide known mathematical theorems that might be relevant"
        )
        
        known_theorems = []
        if theorems_input.strip():
            try:
                known_theorems = json.loads(theorems_input)
            except json.JSONDecodeError:
                st.error("Invalid JSON format for theorems")
        
        # Analysis button
        if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
            run_reasoning_analysis(data, known_theorems, analysis_depth, enable_multi_agent, timeout_seconds)
        
        # Quick analysis buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⚡ Quick Symbolic", use_container_width=True):
                run_symbolic_analysis(data)
        with col2:
            if st.button("🔍 Pattern Search", use_container_width=True):
                run_pattern_analysis(data)
        with col3:
            if st.button("📊 Statistical Summary", use_container_width=True):
                show_statistical_summary(data)

def get_data_input(input_method: str) -> Optional[List[float]]:
    """Get data based on selected input method"""
    
    if input_method == "Manual Entry":
        st.subheader("✏️ Manual Data Entry")
        
        # Option 1: Text input
        data_text = st.text_area(
            "Enter numerical data (comma-separated)",
            placeholder="1, 2, 3, 5, 8, 13, 21, 34",
            help="Enter your data points separated by commas"
        )
        
        if data_text.strip():
            try:
                data = [float(x.strip()) for x in data_text.split(',') if x.strip()]
                return data
            except ValueError:
                st.error("Please enter valid numbers separated by commas")
                return None
        
        # Option 2: Number input with builder
        st.subheader("🔢 Data Builder")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'manual_data' not in st.session_state:
                st.session_state.manual_data = []
            
            new_value = st.number_input("Add value", value=0.0, step=0.1)
            if st.button("➕ Add Value"):
                st.session_state.manual_data.append(new_value)
                st.rerun()
        
        with col2:
            if st.session_state.manual_data:
                st.write("Current data:", st.session_state.manual_data)
                if st.button("🗑️ Clear All"):
                    st.session_state.manual_data = []
                    st.rerun()
                return st.session_state.manual_data
        
        return None
    
    elif input_method == "CSV Upload":
        st.subheader("📁 CSV File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with numerical data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                # Column selection
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    selected_column = st.selectbox("Select data column", numeric_columns)
                    return df[selected_column].dropna().tolist()
                else:
                    st.error("No numeric columns found in the CSV file")
                    return None
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return None
    
    elif input_method == "Example Datasets":
        st.subheader("📊 Example Datasets")
        
        examples = {
            "Linear Sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Quadratic Sequence": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
            "Fibonacci Sequence": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
            "Prime Numbers": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "Exponential Growth": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "Sine Wave": [np.sin(x/2) for x in range(20)],
            "Random Walk": np.cumsum(np.random.randn(50)).tolist()
        }
        
        selected_example = st.selectbox("Choose an example dataset", list(examples.keys()))
        
        if st.button("Load Example Data"):
            return examples[selected_example]
        
        # Show preview
        if selected_example:
            st.write(f"Preview of {selected_example}:")
            preview_data = examples[selected_example][:10]
            st.write(preview_data)
    
    return None

def run_reasoning_analysis(data: List[float], known_theorems: List[Dict], 
                          analysis_depth: str, enable_multi_agent: bool, timeout_seconds: int):
    """Run complete reasoning analysis"""
    
    with st.spinner("🧠 Running AI reasoning analysis..."):
        try:
            # Prepare request
            payload = {
                "data": data,
                "known_theorems": known_theorems,
                "analysis_depth": analysis_depth,
                "enable_multi_agent": enable_multi_agent,
                "timeout_seconds": timeout_seconds
            }
            
            # Make API request
            start_time = time.time()
            response = requests.post(
                f"{st.session_state.api_base_url}/reason",
                json=payload,
                timeout=timeout_seconds + 10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.reasoning_results = result
                
                # Add to history
                st.session_state.analysis_history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "data_points": len(data),
                    "analysis_depth": analysis_depth,
                    "execution_time": end_time - start_time,
                    "quality_score": result.get("quality_score", 0),
                    "results": result
                })
                
                # Display results
                display_reasoning_results(result)
                
            else:
                error_msg = response.json().get("error", "Unknown error") if response.content else "API request failed"
                st.error(f"❌ Analysis failed: {error_msg}")
                
        except requests.TimeoutError:
            st.error("⏱️ Analysis timed out. Try reducing the timeout or data size.")
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def display_reasoning_results(result: Dict[str, Any]):
    """Display comprehensive reasoning results"""
    
    st.success("✅ Analysis completed successfully!")
    
    # Overview metrics
    st.subheader("📊 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Quality Score", f"{result.get('quality_score', 0):.2f}", delta=None)
    
    with col2:
        st.metric("Execution Time", f"{result.get('execution_time', 0):.2f}s")
    
    with col3:
        warnings_count = len(result.get('warnings', []))
        st.metric("Warnings", warnings_count, delta=None)
    
    with col4:
        analysis_depth = result.get('analysis_depth', 'unknown')
        st.metric("Analysis Depth", analysis_depth)
    
    # Final recommendation
    if 'final_recommendation' in result.get('results', {}):
        rec = result['results']['final_recommendation']
        st.subheader("🎯 Final Recommendation")
        
        confidence = rec.get('confidence', 0)
        strength = rec.get('strength', 'UNKNOWN')
        
        if confidence >= 0.8:
            box_class = "success-box"
        elif confidence >= 0.5:
            box_class = "warning-box"
        else:
            box_class = "error-box"
        
        st.markdown(f"""
        <div class="{box_class}">
            <strong>Hypothesis:</strong> {rec.get('hypothesis', 'No hypothesis generated')}<br>
            <strong>Confidence:</strong> {confidence:.2f} ({strength})<br>
            <strong>Best Approach:</strong> {rec.get('supporting_evidence', {}).get('best_approach', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
    
    # Multi-agent results
    if 'multi_agent_analysis' in result.get('results', {}):
        ma_results = result['results']['multi_agent_analysis']
        st.subheader("🤖 Multi-Agent Analysis")
        
        # Agent results summary
        agent_results = ma_results.get('multi_agent_results', {})
        
        for agent_name, agent_result in agent_results.items():
            with st.expander(f"🔍 {agent_name.replace('_', ' ').title()}"):
                if hasattr(agent_result, '__dict__'):
                    st.json(agent_result.__dict__)
                else:
                    st.json(agent_result)
    
    # Symbolic regression results
    if 'symbolic_regression' in result.get('results', {}):
        sr_results = result['results']['symbolic_regression']
        st.subheader("🔬 Symbolic Regression")
        
        best_candidate = sr_results.get('best_candidate')
        if best_candidate:
            expr, metrics = best_candidate
            st.success(f"**Best Expression:** `{expr}`")
            st.info(f"**Score:** {metrics.final_score:.3f} | **R²:** {metrics.r_squared:.3f}")
        
        # Strategy comparison
        strategy_results = sr_results.get('strategy_results', {})
        if strategy_results:
            st.subheader("Strategy Comparison")
            
            strategy_data = []
            for strategy, results in strategy_results.items():
                if isinstance(results, dict) and 'best_score' in results:
                    strategy_data.append({
                        'Strategy': strategy.title(),
                        'Best Score': results['best_score'],
                        'Candidates': results.get('candidates_generated', 0)
                    })
            
            if strategy_data:
                df = pd.DataFrame(strategy_data)
                st.dataframe(df, use_container_width=True)
    
    # Proof verification
    if 'proof_verification' in result.get('results', {}):
        proof_results = result['results']['proof_verification']
        st.subheader("📜 Proof Verification")
        
        if proof_results.get('is_provable'):
            st.success("✅ Formal proof verified!")
            st.code(proof_results.get('formal_expression', ''), language='lean')
        else:
            st.warning("⚠️ Proof verification failed or not attempted")
            if 'error' in proof_results:
                st.error(proof_results['error'])
    
    # Warnings and quality assessment
    warnings = result.get('warnings', [])
    if warnings:
        st.subheader("⚠️ Warnings")
        for warning in warnings:
            st.warning(warning)

def run_symbolic_analysis(data: List[float]):
    """Run quick symbolic analysis"""
    
    with st.spinner("⚡ Running symbolic regression..."):
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/analyze/symbolic",
                json=data,
                params={"strategy": "mixed"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("⚡ Quick Symbolic Analysis Results")
                
                top_candidates = result.get('top_candidates', [])
                if top_candidates:
                    df = pd.DataFrame(top_candidates)
                    df['score'] = df['score'].round(3)
                    df['r_squared'] = df['r_squared'].round(3)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No good candidates found")
            else:
                st.error("Symbolic analysis failed")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def run_pattern_analysis(data: List[float]):
    """Run pattern analysis"""
    st.subheader("🔍 Pattern Analysis")
    
    # Basic pattern detection
    differences = [data[i+1] - data[i] for i in range(len(data)-1)]
    second_differences = [differences[i+1] - differences[i] for i in range(len(differences)-1)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First Differences:**")
        st.write(differences)
        
        if len(set(differences)) == 1:
            st.success("Constant differences detected - likely linear!")
    
    with col2:
        st.write("**Second Differences:**")
        st.write(second_differences)
        
        if len(set(second_differences)) == 1:
            st.success("Constant second differences - likely quadratic!")

def show_statistical_summary(data: List[float]):
    """Show statistical summary"""
    st.subheader("📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Descriptive Statistics:**")
        stats = {
            "Count": len(data),
            "Mean": np.mean(data),
            "Median": np.median(data),
            "Std Dev": np.std(data),
            "Min": np.min(data),
            "Max": np.max(data),
            "Range": np.max(data) - np.min(data)
        }
        
        for key, value in stats.items():
            st.metric(key, f"{value:.3f}" if isinstance(value, float) else value)
    
    with col2:
        st.write("**Distribution:**")
        fig, ax = plt.subplots()
        ax.hist(data, bins=min(20, len(data)//2 + 1), alpha=0.7)
        ax.set_title("Data Distribution")
        st.pyplot(fig)

def run_visualization_tab():
    """Visualization tab"""
    st.header("📊 Data Visualization & Analysis")
    
    if st.session_state.reasoning_results is None:
        st.info("👆 Run an analysis first to see visualizations")
        return
    
    # Implementation for advanced visualizations
    st.write("Advanced visualizations coming soon...")

def run_proof_explorer_tab():
    """Proof explorer tab"""
    st.header("📜 Formal Proof Explorer")
    
    st.subheader("🔍 Proof Verification")
    
    formal_expr = st.text_area(
        "Enter formal expression (Lean 4 syntax)",
        placeholder="theorem simple : 2 + 2 = 4 := by ring",
        help="Enter a mathematical statement in Lean 4 format"
    )
    
    if st.button("🔎 Verify Proof") and formal_expr.strip():
        with st.spinner("Verifying proof..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_base_url}/prove",
                    json=formal_expr,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['is_valid']:
                        st.success("✅ Proof verified!")
                    else:
                        st.error("❌ Proof verification failed")
                    
                    st.json(result)
                else:
                    st.error("Proof verification request failed")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def run_history_tab():
    """Analysis history tab"""
    st.header("📈 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Run some analyses to see them here!")
        return
    
    # Display history
    for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}: {analysis['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Points", analysis['data_points'])
                st.metric("Quality Score", f"{analysis['quality_score']:.2f}")
            
            with col2:
                st.metric("Execution Time", f"{analysis['execution_time']:.2f}s")
                st.metric("Analysis Depth", analysis['analysis_depth'])
            
            with col3:
                if st.button(f"🔄 Reload Analysis {len(st.session_state.analysis_history) - i}"):
                    st.session_state.reasoning_results = analysis['results']
                    st.rerun()

if __name__ == "__main__":
    main()
