"""Callable Cosmos SDK example; supply your own local media (none is fabricated)."""
import argparse
import json
from edgeforge_sdk import CosmosReason


def main():
    parser = argparse.ArgumentParser(description="EdgeForge Cosmos/Qwen3-VL generation")
    parser.add_argument("--model", default="model")
    parser.add_argument("--prompt", required=True)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--image")
    media.add_argument("--video")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--fps", type=float, help="Requested sampling FPS, not source FPS")
    sampling.add_argument("--num-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, help="Per-image / per-video-temporal-group resolution budget")
    parser.add_argument("--video-total-pixels", type=int, help="Explicit HF video total T*H*W budget")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--greedy", action="store_true", help="Override original generation_config.do_sample")
    parser.add_argument("--seed", type=int)
    args = vars(parser.parse_args())
    model_path = args.pop("model")
    generation = {}
    if args.pop("greedy"):
        generation["do_sample"] = False
    seed = args.pop("seed")
    if seed is not None:
        generation["seed"] = seed
    with CosmosReason(model_path) as model:
        result = model.generate(generation=generation, **args)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
