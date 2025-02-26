from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from stepvideo.config import parse_args
from stepvideo.parallel import initialize_parall_group, get_parallel_group
from stepvideo.utils import setup_seed
from xfuser.model_executor.models.customized.step_video_t2v.tp_applicator import TensorParallelApplicator
from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

if __name__ == "__main__":
    args = parse_args()
    initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree, tensor_parallel_degree=args.tensor_parallel_degree)
    
    local_rank = get_parallel_group().local_rank
    device = torch.device(f"cuda:{local_rank}")
    
    setup_seed(args.seed)
        
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16, device="cpu")

    if args.tensor_parallel_degree > 1:
        tp_applicator = TensorParallelApplicator(get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank())
        tp_applicator.apply_to_model(pipeline.transformer)

    pipeline.transformer = pipeline.transformer.to(device)
    pipeline.setup_api(
        vae_url = args.vae_url,
        caption_url = args.caption_url,
    )
    
    
    prompt = args.prompt
    videos = pipeline(
        prompt=prompt, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps = args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50]
    )
    
    dist.destroy_process_group()