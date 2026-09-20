#!/usr/bin/env python
"""
Benchmark script for AraSpellX training pipeline.
Measures data loading, forward, backward, optimizer, and total step times.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.args import get_train_args, validate_training_args
from core.train import get_trainer
from core.utils import load_state


def benchmark_step(trainer, num_steps=100, warmup=5):
    """Benchmark a number of training steps."""
    device = trainer.device
    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    train_loader = trainer.train_loader
    
    # Set to train mode
    model.train()
    
    # Timing accumulators
    data_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []
    
    # Peak memory tracking
    peak_mem = 0
    
    print(f"Running benchmark: {num_steps} steps (warmup: {warmup})")
    print(f"Device: {device}")
    print(f"Batch size: {trainer.train_loader.batch_size}")
    print("-" * 60)
    
    step_count = 0
    for batch_idx, batch in enumerate(train_loader):
        if step_count >= num_steps + warmup:
            break
            
        # Data loading time
        data_start = time.perf_counter()
        (enc_inp, dec_inp, enc_mask, dec_mask) = batch
        enc_inp = enc_inp.to(device, non_blocking=False)
        dec_inp = dec_inp.to(device, non_blocking=False)
        enc_mask = enc_mask.to(device, non_blocking=False)
        dec_mask = dec_mask.to(device, non_blocking=False)
        data_time = time.perf_counter() - data_start
        
        if step_count >= warmup:
            data_times.append(data_time)
        
        # Forward pass
        torch.cuda.synchronize()
        fwd_start = time.perf_counter()
        optimizer.zero_grad()
        preds, att = model(enc_inp, dec_inp, enc_mask, dec_mask)
        loss = criterion(preds, dec_inp, dec_mask)
        fwd_time = time.perf_counter() - fwd_start
        
        if step_count >= warmup:
            forward_times.append(fwd_time)
        
        # Backward pass
        torch.cuda.synchronize()
        bwd_start = time.perf_counter()
        loss.backward()
        bwd_time = time.perf_counter() - bwd_start
        
        if step_count >= warmup:
            backward_times.append(bwd_time)
        
        # Optimizer step
        torch.cuda.synchronize()
        opt_start = time.perf_counter()
        optimizer.step()
        opt_time = time.perf_counter() - opt_start
        
        if step_count >= warmup:
            optimizer_times.append(opt_time)
        
        # Total step time
        torch.cuda.synchronize()
        if step_count >= warmup:
            total_time = data_time + fwd_time + bwd_time + opt_time
            total_times.append(total_time)
            
            # Track peak memory
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024**3)
        
        step_count += 1
        
        if step_count % 10 == 0 and step_count > warmup:
            avg_total = sum(total_times) / len(total_times)
            print(f"Step {step_count - warmup}/{num_steps}: "
                  f"total={avg_total:.4f}s, "
                  f"data={sum(data_times)/len(data_times):.4f}s, "
                  f"fwd={sum(forward_times)/len(forward_times):.4f}s, "
                  f"bwd={sum(backward_times)/len(backward_times):.4f}s, "
                  f"opt={sum(optimizer_times)/len(optimizer_times):.4f}s")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    n = len(total_times)
    if n == 0:
        print("No steps measured!")
        return
    
    avg_data = sum(data_times) / n
    avg_fwd = sum(forward_times) / n
    avg_bwd = sum(backward_times) / n
    avg_opt = sum(optimizer_times) / n
    avg_total = sum(total_times) / n
    
    batch_size = trainer.train_loader.batch_size
    samples_per_sec = batch_size / avg_total
    tokens_per_sec = samples_per_sec * trainer.train_loader.dataset.max_len * 2  # enc + dec
    
    print(f"Steps measured: {n}")
    print(f"Batch size: {batch_size}")
    print(f"Avg data loading: {avg_data*1000:.2f} ms")
    print(f"Avg forward pass: {avg_fwd*1000:.2f} ms")
    print(f"Avg backward pass: {avg_bwd*1000:.2f} ms")
    print(f"Avg optimizer step: {avg_opt*1000:.2f} ms")
    print(f"Avg total step: {avg_total*1000:.2f} ms")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Tokens/sec (est): {tokens_per_sec:.0f}")
    print(f"Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Estimated time per epoch (6.9M samples): {6922318 / samples_per_sec / 3600:.2f} hours")
    print("=" * 60)
    
    return {
        'steps': n,
        'batch_size': batch_size,
        'avg_data_ms': avg_data * 1000,
        'avg_fwd_ms': avg_fwd * 1000,
        'avg_bwd_ms': avg_bwd * 1000,
        'avg_opt_ms': avg_opt * 1000,
        'avg_total_ms': avg_total * 1000,
        'samples_per_sec': samples_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'peak_gpu_mem_gb': peak_mem,
    }


def benchmark_with_profiler(trainer, num_steps=10, output_dir="profile_out"):
    """Run benchmark with PyTorch profiler."""
    os.makedirs(output_dir, exist_ok=True)
    
    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    train_loader = trainer.train_loader
    device = trainer.device
    
    model.train()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
    ) as prof:
        for step, batch in enumerate(train_loader):
            if step >= num_steps:
                break
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            enc_inp = enc_inp.to(device)
            dec_inp = dec_inp.to(device)
            enc_mask = enc_mask.to(device)
            dec_mask = dec_mask.to(device)
            
            optimizer.zero_grad()
            preds, att = model(enc_inp, dec_inp, enc_mask, dec_mask)
            loss = criterion(preds, dec_inp, dec_mask)
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                print(f"Profiled step {step}/{num_steps}")
    
    print(f"\nProfiler trace saved to {output_dir}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description="Benchmark AraSpellX training")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps")
    parser.add_argument("--profile", action="store_true", help="Run with PyTorch profiler")
    parser.add_argument("--profile-steps", type=int, default=10, help="Steps for profiler")
    parser.add_argument("--profile-dir", type=str, default="profile_out", help="Profiler output dir")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs (for config)")
    # Parse known args to separate benchmark args from training args
    bench_args, remaining_argv = parser.parse_known_args()
    
    # Reconstruct sys.argv for training args parser
    sys.argv = [sys.argv[0]] + remaining_argv
    # Ensure epochs is set
    if '--epochs' not in remaining_argv:
        sys.argv += ['--epochs', str(bench_args.epochs)]
    
    args = bench_args
    
    print("=" * 60)
    print("AraSpellX Training Benchmark")
    print("=" * 60)
    
    # Get training args
    train_args = get_train_args()
    train_args = validate_training_args(train_args)
    
    # Print config
    print(f"\nConfiguration:")
    print(f"  epochs: {train_args.epochs}")
    print(f"  batch_size: {train_args.batch_size}")
    print(f"  max_len: {train_args.max_len}")
    print(f"  distortion_ratio: {train_args.distortion_ratio}")
    print(f"  d_model: {train_args.d_model}")
    print(f"  n_layers: {train_args.n_layers}")
    print(f"  h: {train_args.h}")
    print(f"  hidden_size: {train_args.hidden_size}")
    print(f"  clip_grad: {train_args.clip_grad}")
    print(f"  grad_norm: {train_args.grad_norm}")
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = get_trainer(rank=0, args=train_args)
    
    if args.profile:
        print(f"\nRunning profiler for {args.profile_steps} steps...")
        benchmark_with_profiler(trainer, args.profile_steps, args.profile_dir)
    else:
        print(f"\nRunning benchmark for {args.steps} steps...")
        benchmark_step(trainer, args.steps, args.warmup)


if __name__ == "__main__":
    main()