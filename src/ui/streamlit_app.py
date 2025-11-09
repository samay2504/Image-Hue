"""
Streamlit UI for image colorization.

Features:
- Image upload and example images
- Method selector (classification, L2, OpenCV)
- Temperature slider with animation
- Blend animation (grayscale to colored)
- Color distribution visualization
- Model info panel
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import base64
import time

# Import colorization modules
from src.infer import ColorizationInference
from src.models.ops import DEFAULT_TEMPERATURE

# Page config
st.set_page_config(
    page_title="Colorful Image Colorization",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'colorized_result' not in st.session_state:
    st.session_state.colorized_result = None
if 'blend_frames' not in st.session_state:
    st.session_state.blend_frames = None


@st.cache_resource
def load_inference_engine(model_path=None, redis_url=None):
    """Load and cache inference engine."""
    return ColorizationInference(
        model_path=model_path,
        device=None,  # Auto-detect
        use_cache=True,
        redis_url=redis_url
    )


def create_download_link(img_array, filename="colorized.png"):
    """Create download link for image."""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">💾 Download Image</a>'


def display_blend_animation(frames, fps=20):
    """Display blend animation using Streamlit's native image display."""
    animation_placeholder = st.empty()
    
    # Calculate delay
    delay = 1.0 / fps
    
    # Loop through frames
    for frame in frames:
        frame_img = Image.fromarray((frame * 255).astype(np.uint8))
        animation_placeholder.image(frame_img, use_container_width=True)
        time.sleep(delay)
    
    # Show final frame
    animation_placeholder.image(
        Image.fromarray((frames[-1] * 255).astype(np.uint8)),
        use_container_width=True
    )


def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Colorful Image Colorization</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Transform grayscale images into vibrant colors using deep learning!</p>
        <p style="font-size: 0.9rem; color: #666;">
            Based on "Colorful Image Colorization" (Zhang, Isola, Efros - ECCV 2016)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        model_path = st.text_input("Model Checkpoint Path (optional)", "")
        redis_url = st.text_input("Redis URL (optional)", "redis://localhost:6379")
        
        if st.button("🔄 Initialize/Reload Model"):
            with st.spinner("Loading model..."):
                st.session_state.inference_engine = load_inference_engine(
                    model_path if model_path else None,
                    redis_url if redis_url else None
                )
                st.success("Model loaded!")
        
        st.divider()
        
        # Method selector
        st.subheader("🎯 Colorization Method")
        method = st.selectbox(
            "Select method",
            ["classification", "l2", "opencv"],
            format_func=lambda x: {
                "classification": "📊 Paper Classification (Recommended)",
                "l2": "📐 L2 Regression Baseline",
                "opencv": "🔧 OpenCV Color Transfer"
            }[x]
        )
        
        # Temperature slider (only for classification)
        temperature = DEFAULT_TEMPERATURE
        if method == "classification":
            st.subheader("🌡️ Temperature")
            temperature = st.slider(
                "Annealed-mean temperature",
                min_value=0.01,
                max_value=1.0,
                value=DEFAULT_TEMPERATURE,
                step=0.01,
                help="Lower values = more vibrant, Higher values = more conservative"
            )
        
        # Blend slider
        st.subheader("🎬 Animation")
        num_blend_frames = st.slider(
            "Animation frames",
            min_value=10,
            max_value=60,
            value=30,
            step=5
        )
        
        # Model info
        st.divider()
        st.subheader("📊 System Info")
        
        import torch
        if torch.cuda.is_available():
            from src.utils.memory import get_gpu_memory_info
            alloc, reserved, free = get_gpu_memory_info()
            st.metric("GPU Memory", f"{alloc:.2f} GB used")
            st.metric("Free Memory", f"{free:.2f} GB")
        else:
            st.info("Running on CPU")
        
        # Cache stats
        if st.session_state.inference_engine and st.session_state.inference_engine.cache:
            cache_stats = st.session_state.inference_engine.cache.get_stats()
            st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']*100:.1f}%")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input Image")
        
        # Image upload
        upload_tab, example_tab = st.tabs(["Upload", "Examples"])
        
        with upload_tab:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Upload a grayscale or color image to colorize"
            )
            
            if uploaded_file:
                img = Image.open(uploaded_file).convert('RGB')
                st.session_state.current_image = np.array(img) / 255.0
                st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with example_tab:
            # Example images
            examples_dir = Path("examples")
            if examples_dir.exists():
                example_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
                if example_files:
                    example_choice = st.selectbox("Select example", example_files)
                    if st.button("Load Example"):
                        img = Image.open(example_choice).convert('RGB')
                        st.session_state.current_image = np.array(img) / 255.0
                        st.image(img, caption="Example Image", use_container_width=True)
                else:
                    st.info("No example images found in examples/ directory")
            else:
                st.info("Create an examples/ directory and add sample images!")
        
        # Colorize button
        if st.session_state.current_image is not None:
            if st.button("🎨 Colorize!", type="primary", use_container_width=True):
                # Initialize engine if not loaded
                if st.session_state.inference_engine is None:
                    with st.spinner("Initializing model..."):
                        st.session_state.inference_engine = load_inference_engine()
                
                # Colorize
                with st.spinner(f"Colorizing with {method} method..."):
                    try:
                        result = st.session_state.inference_engine.colorize_image(
                            st.session_state.current_image,
                            method=method,
                            temperature=temperature
                        )
                        st.session_state.colorized_result = result
                        st.success("✅ Colorization complete!")
                    except Exception as e:
                        st.error(f"Error during colorization: {e}")
    
    with col2:
        st.header("🎨 Colorized Output")
        
        if st.session_state.colorized_result is not None:
            result_img = Image.fromarray((st.session_state.colorized_result * 255).astype(np.uint8))
            st.image(result_img, caption="Colorized Result", use_container_width=True)
            
            # Download button
            st.markdown(
                create_download_link(st.session_state.colorized_result),
                unsafe_allow_html=True
            )
            
            # Comparison slider (side-by-side)
            st.subheader("📊 Comparison")
            compare_cols = st.columns(2)
            with compare_cols[0]:
                # Grayscale version
                gray = np.mean(st.session_state.current_image, axis=2, keepdims=True)
                gray_rgb = np.repeat(gray, 3, axis=2)
                st.image(gray_rgb, caption="Grayscale", use_container_width=True)
            with compare_cols[1]:
                st.image(result_img, caption="Colorized", use_container_width=True)
        else:
            st.info("👈 Upload an image and click 'Colorize!' to see results")
    
    # Blend animation section
    if st.session_state.colorized_result is not None:
        st.divider()
        st.header("🎬 Blend Animation")
        
        col_anim1, col_anim2 = st.columns([3, 1])
        
        with col_anim2:
            if st.button("▶️ Generate Animation", use_container_width=True):
                with st.spinner("Creating animation..."):
                    frames = st.session_state.inference_engine.create_blend_animation(
                        st.session_state.current_image,
                        method=method,
                        temperature=temperature,
                        num_frames=num_blend_frames
                    )
                    st.session_state.blend_frames = frames
            
            if st.session_state.blend_frames:
                if st.button("🔄 Play Animation", use_container_width=True):
                    with col_anim1:
                        display_blend_animation(st.session_state.blend_frames, fps=20)
        
        with col_anim1:
            if st.session_state.blend_frames:
                # Show blend slider
                blend_ratio = st.slider(
                    "Blend ratio (0 = grayscale, 100 = full color)",
                    0, 100, 100, 1
                )
                
                # Compute blended frame
                gray = np.mean(st.session_state.current_image, axis=2, keepdims=True)
                gray_rgb = np.repeat(gray, 3, axis=2)
                
                alpha = blend_ratio / 100.0
                blended = gray_rgb * (1 - alpha) + st.session_state.colorized_result * alpha
                
                st.image(blended, caption=f"Blend: {blend_ratio}%", use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎨 Colorful Image Colorization | Built with Streamlit</p>
        <p style="font-size: 0.8rem;">
            Implementation of Zhang, Isola, Efros (ECCV 2016)<br>
            "Colorful Image Colorization"
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
