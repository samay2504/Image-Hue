"""
Gradio UI for image colorization.

Features:
- Image upload and examples
- Method selector
- Temperature slider
- Blend animation
- Model info
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gradio as gr
import numpy as np
from PIL import Image
import os

from src.infer import ColorizationInference
from src.models.ops import DEFAULT_TEMPERATURE


class ColorizationUI:
    """Gradio UI for colorization."""
    
    def __init__(self, model_path=None, redis_url=None):
        self.engine = ColorizationInference(
            model_path=model_path,
            device=None,
            use_cache=True,
            redis_url=redis_url
        )
    
    def colorize(self, image, method, temperature):
        """Colorize image."""
        if image is None:
            return None, "Please upload an image first!"
        
        try:
            # Colorize
            result = self.engine.colorize_image(
                image,
                method=method,
                temperature=temperature
            )
            
            # Convert to uint8 for display
            result_img = (result * 255).astype(np.uint8)
            
            info = f"✅ Colorization complete using {method} method"
            if method == "classification":
                info += f" (T={temperature:.2f})"
            
            return result_img, info
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_comparison(self, image, colorized):
        """Create side-by-side comparison."""
        if image is None or colorized is None:
            return None
        
        # Convert to grayscale
        if isinstance(image, np.ndarray):
            gray = np.mean(image, axis=2, keepdims=True)
            gray = np.repeat(gray, 3, axis=2).astype(np.uint8)
        else:
            gray = np.array(Image.fromarray(image).convert('L').convert('RGB'))
        
        # Concatenate horizontally
        comparison = np.concatenate([gray, colorized], axis=1)
        return comparison
    
    def create_blend_animation(self, image, method, temperature, num_frames):
        """Create blend animation frames."""
        if image is None:
            return None, "Please upload an image first!"
        
        try:
            frames = self.engine.create_blend_animation(
                image,
                method=method,
                temperature=temperature,
                num_frames=num_frames
            )
            
            # Convert to uint8
            frames_uint8 = [(f * 255).astype(np.uint8) for f in frames]
            
            return frames_uint8, f"✅ Generated {len(frames)} animation frames"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def get_blended_frame(self, image, colorized, blend_ratio):
        """Get single blended frame based on slider."""
        if image is None or colorized is None:
            return None
        
        # Convert to grayscale
        gray = np.mean(image, axis=2, keepdims=True)
        gray = np.repeat(gray, 3, axis=2)
        
        # Blend
        alpha = blend_ratio / 100.0
        if isinstance(colorized, np.ndarray):
            colorized_float = colorized.astype(np.float32) / 255.0
        else:
            colorized_float = colorized
        
        blended = gray * (1 - alpha) + colorized_float * alpha
        return (blended * 255).astype(np.uint8)
    
    def build_interface(self):
        """Build Gradio interface."""
        
        with gr.Blocks(title="Colorful Image Colorization", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🎨 Colorful Image Colorization
            
            Transform grayscale images into vibrant colors using deep learning!
            
            *Based on "Colorful Image Colorization" (Zhang, Isola, Efros - ECCV 2016)*
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Input")
                    
                    # Image input
                    input_image = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        sources=["upload", "clipboard"]
                    )
                    
                    # Method selector
                    method = gr.Radio(
                        choices=["classification", "l2", "opencv"],
                        value="classification",
                        label="Colorization Method",
                        info="Classification = paper method (recommended)"
                    )
                    
                    # Temperature slider (only visible for classification)
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=DEFAULT_TEMPERATURE,
                        step=0.01,
                        label="Temperature (T)",
                        info="Lower = more vibrant, Higher = more conservative"
                    )
                    
                    # Colorize button
                    colorize_btn = gr.Button("🎨 Colorize!", variant="primary", size="lg")
                    
                    # Status
                    status_text = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎨 Output")
                    
                    # Output image
                    output_image = gr.Image(label="Colorized Result", type="numpy")
                    
                    # Comparison
                    comparison_image = gr.Image(label="Comparison (Grayscale | Colorized)")
            
            # Blend animation section
            gr.Markdown("---")
            gr.Markdown("### 🎬 Blend Animation")
            
            with gr.Row():
                with gr.Column(scale=1):
                    num_frames = gr.Slider(
                        minimum=10,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Number of Frames"
                    )
                    
                    animate_btn = gr.Button("▶️ Generate Animation", variant="secondary")
                    
                    blend_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=100,
                        step=1,
                        label="Blend Ratio (0=grayscale, 100=full color)"
                    )
                    
                    anim_status = gr.Textbox(label="Animation Status", interactive=False)
                
                with gr.Column(scale=1):
                    # Blended frame display
                    blended_output = gr.Image(label="Blended Frame")
                    
                    # Animation gallery (show all frames)
                    animation_gallery = gr.Gallery(
                        label="Animation Frames",
                        columns=6,
                        height="auto"
                    )
            
            # System info
            gr.Markdown("---")
            gr.Markdown("### 📊 System Info")
            
            with gr.Row():
                import torch
                if torch.cuda.is_available():
                    from src.utils.memory import get_gpu_memory_info
                    alloc, reserved, free = get_gpu_memory_info()
                    gr.Markdown(f"""
                    - **Device**: CUDA GPU
                    - **GPU Memory Used**: {alloc:.2f} GB
                    - **GPU Memory Free**: {free:.2f} GB
                    """)
                else:
                    gr.Markdown("- **Device**: CPU")
                
                # Cache stats
                if self.engine.cache:
                    cache_stats = self.engine.cache.get_stats()
                    gr.Markdown(f"""
                    - **Cache Hits**: {cache_stats['hits']}
                    - **Cache Misses**: {cache_stats['misses']}
                    - **Hit Rate**: {cache_stats['hit_rate']*100:.1f}%
                    """)
            
            # Event handlers
            colorize_btn.click(
                fn=self.colorize,
                inputs=[input_image, method, temperature],
                outputs=[output_image, status_text]
            ).then(
                fn=self.create_comparison,
                inputs=[input_image, output_image],
                outputs=[comparison_image]
            )
            
            animate_btn.click(
                fn=self.create_blend_animation,
                inputs=[input_image, method, temperature, num_frames],
                outputs=[animation_gallery, anim_status]
            )
            
            blend_slider.change(
                fn=self.get_blended_frame,
                inputs=[input_image, output_image, blend_slider],
                outputs=[blended_output]
            )
            
            # Examples
            examples_dir = Path("examples")
            if examples_dir.exists():
                example_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
                if example_files:
                    gr.Examples(
                        examples=[[str(f)] for f in example_files[:5]],
                        inputs=[input_image],
                        label="Example Images"
                    )
        
        return demo


def launch_gradio_app(model_path=None, redis_url=None, share=False, server_port=7860):
    """Launch Gradio app."""
    ui = ColorizationUI(model_path=model_path, redis_url=redis_url)
    demo = ui.build_interface()
    
    demo.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model checkpoint path")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    
    args = parser.parse_args()
    
    launch_gradio_app(
        model_path=args.model,
        redis_url=args.redis_url,
        share=args.share,
        server_port=args.port
    )
