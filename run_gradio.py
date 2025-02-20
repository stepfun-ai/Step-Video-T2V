from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch
import gradio as gr
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
import os


def initialize_pipeline(model_dir, vae_url=None, caption_url=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = StepVideoPipeline.from_pretrained(model_dir).to(
        dtype=torch.bfloat16, device=device
    )
    if vae_url or caption_url:
        pipeline.setup_api(vae_url=vae_url, caption_url=caption_url)
    return pipeline


def generate_video(
    prompt,
    model_dir,
    num_frames=204,
    height=544,
    width=992,
    num_inference_steps=50,
    guidance_scale=9.0,
    time_shift=7.0,
    pos_magic="超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。",
    neg_magic="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。",
    seed=1234,
):
    setup_seed(seed)
    pipeline = initialize_pipeline(model_dir)

    output = pipeline(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        time_shift=time_shift,
        pos_magic=pos_magic,
        neg_magic=neg_magic,
        output_file_name=prompt[:50],
    )

    if isinstance(output.video, torch.Tensor):
        video = output.video.cpu().numpy()
    else:
        video = output.video

    # Ensure video is in the correct format for Gradio (T, H, W, C)
    if len(video.shape) == 5:  # (B, T, C, H, W)
        video = video[0].transpose(0, 2, 3, 1)  # (T, H, W, C)
    elif len(video.shape) == 4 and video.shape[-1] != 3:  # (T, C, H, W)
        video = video.transpose(0, 2, 3, 1)  # (T, H, W, C)

    return video


def create_interface():
    MODEL_CONFIGS = {
        "stepfun-ai/stepvideo-t2v": {
            "max_frames": 204,
            "default_frames": 204,
            "cfg_scale": 9.0,
            "time_shift": 13.0,
            "inference_steps": 50
        },
        "stepfun-ai/stepvideo-t2v-turbo": {
            "max_frames": 136,
            "default_frames": 136,
            "cfg_scale": 5.0,
            "time_shift": 17.0,
            "inference_steps": 15
        }
    }

    def update_frames(model):
        config = MODEL_CONFIGS[model]
        return [
            gr.update(maximum=config["max_frames"], value=config["default_frames"]),
            gr.update(value=config["cfg_scale"]),
            gr.update(value=config["time_shift"]),
            gr.update(value=config["inference_steps"])
        ]

    with gr.Blocks(
        theme=gr.themes.Glass(
            primary_hue="green",
            secondary_hue="violet",
            neutral_hue="slate",
        )
    ) as demo:
        gr.Markdown(
            """
            # Step-Video-T2V
            a state-of-the-art (SoTA) text-to-video pre-trained model with 30 billion parameters and the capability to generate videos up to 204 frames.

            <div align="center">
                <strong>Step-Video Team</strong>
            </div>
            <div align="center" style="display: flex; justify-content: center; gap: 5px; margin-bottom: 10px;">
                <a href="https://yuewen.cn/videos"><img src="https://img.shields.io/static/v1?label=Step-Video&message=Web&color=green"></a>
                <a href=""><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a>
                <a href="https://x.com/StepFun_ai"><img src="https://img.shields.io/static/v1?label=X.com&message=Web&color=blue"></a>
            </div>

            <div align="center" style="display: flex; justify-content: center; gap: 5px;">
                <a href="https://huggingface.co/stepfun-ai/stepvideo-t2v"><img src="https://img.shields.io/static/v1?label=Step-Video-T2V&message=HuggingFace&color=yellow"></a>
                <a href="https://huggingface.co/stepfun-ai/stepvideo-t2v-turbo"><img src="https://img.shields.io/static/v1?label=Step-Video-T2V-Turbo&message=HuggingFace&color=yellow"></a>
            </div>
        """
        )

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=5,
                )
                model_dir = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_CONFIGS.keys()),
                    value="stepfun-ai/stepvideo-t2v",
                )

                with gr.Row():
                    num_frames = gr.Slider(
                        minimum=17,
                        maximum=204,
                        value=204,
                        step=17,
                        label="Number of Frames",
                    )
                    height = gr.Slider(
                        minimum=256, maximum=1024, value=544, step=32, label="Height"
                    )
                    width = gr.Slider(
                        minimum=256, maximum=1024, value=992, step=32, label="Width"
                    )

                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=50,
                        step=1,
                        label="Inference Steps",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=9.0,
                        step=0.1,
                        label="Guidance Scale",
                    )

                with gr.Row():
                    time_shift = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=13.0,
                        step=0.1,
                        label="Time Shift",
                    )
                    pos_magic = gr.Textbox(
                        label="Positive Magic",
                        value="超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。",
                        lines=2
                    )
                    neg_magic = gr.Textbox(
                        label="Negative Magic",
                        value="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。",
                        lines=2
                    )

                seed = gr.Slider(
                    minimum=0, maximum=1000000, value=1234, step=1, label="Random Seed"
                )
                generate_btn = gr.Button("Generate Video")

            with gr.Column():
                output_video = gr.Video(label="Generated Video")

        model_dir.change(fn=update_frames, inputs=[model_dir], outputs=[num_frames, guidance_scale, time_shift, num_inference_steps])

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                model_dir,
                num_frames,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                time_shift,
                pos_magic,
                neg_magic,
                seed,
            ],
            outputs=output_video,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    demo.launch(server_name=host, share=True, inbrowser=True)
