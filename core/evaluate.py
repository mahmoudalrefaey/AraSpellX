import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module

from core.args import get_eval_args
from core.logger import get_logger
from data.data import get_test_loader
from data.tokenizer import get_tokenizer, CharTokenizer
from models.models import get_model, Transformer
from core.utils import save_json


class Evaluator:
    def __init__(
        self,
        model: Module,
        test_loader,
        tokenizer: CharTokenizer,
        device: torch.device,
        max_gen_len: int,
        logger
    ):
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self.max_gen_len = max_gen_len
        self.logger = logger
        
        self.sos_id = tokenizer.special_tokens.sos_id
        self.eos_id = tokenizer.special_tokens.eos_id
        self.pad_id = tokenizer.special_tokens.pad_id

    @torch.inference_mode()
    def generate(
        self,
        enc_inp: Tensor,
        enc_mask: Tensor
    ) -> Tensor:
        """Autoregressive generation for a batch.
        
        Args:
            enc_inp: [B, L_enc] encoder input (distorted text)
            enc_mask: [B, L_enc] encoder mask (True = padding)
            
        Returns:
            predictions: [B, L_dec] generated token IDs (including SOS, excluding EOS/PAD after EOS)
        """
        batch_size = enc_inp.shape[0]
        
        self.model.eval()
        
        # Encode once
        if isinstance(self.model, Transformer):
            enc_out = self.model.encoder(x=enc_inp, mask=enc_mask)
        else:
            raise NotImplementedError("Only Transformer model supported for evaluation")
        
        # Initialize decoder input with SOS token
        dec_inp = torch.full(
            (batch_size, 1), self.sos_id, dtype=torch.long, device=self.device
        )
        
        # Track which sequences have finished (generated EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Store generated tokens
        generated = []
        
        # Causal mask for decoder self-attention
        # Will be rebuilt each step with increasing sequence length
        for step in range(self.max_gen_len):
            # Create causal mask for current decoder length
            dec_len = dec_inp.shape[1]
            causal_mask = torch.triu(
                torch.ones(dec_len, dec_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            causal_mask = ~causal_mask  # True = keep (j <= i)
            dec_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Forward pass through decoder
            if isinstance(self.model, Transformer):
                out, _ = self.model.decoder(
                    x=dec_inp,
                    mask=dec_mask,
                    enc_values=enc_out,
                    key_mask=enc_mask,
                    need_weights=False
                )
                out = self.model.fc(out)
                logits = torch.nn.functional.log_softmax(out, dim=-1)
            else:
                raise NotImplementedError
            
            # Get next token predictions (last position)
            next_token_logits = logits[:, -1, :]  # [B, V]
            next_token = next_token_logits.argmax(dim=-1)  # [B]
            
            # Append to generated
            generated.append(next_token.unsqueeze(1))
            
            # Update finished sequences
            finished = finished | (next_token == self.eos_id)
            
            # If all finished, break
            if finished.all():
                break
            
            # Append next token to decoder input for next step
            dec_inp = torch.cat([dec_inp, next_token.unsqueeze(1)], dim=1)
        
        # Concatenate all generated tokens: [B, L_gen]
        generated = torch.cat(generated, dim=1)
        
        return generated

    def decode_tokens(self, token_ids: Tensor) -> List[str]:
        """Convert token IDs to text, removing special tokens."""
        texts = []
        for seq in token_ids:
            tokens = []
            for tid in seq.tolist():
                if tid == self.eos_id:
                    break
                if tid in (self.sos_id, self.pad_id):
                    continue
                tokens.append(tid)
            text = ''.join(self.tokenizer.ids2tokens(tokens))
            texts.append(text)
        return texts

    def decode_target(self, token_ids: Tensor) -> List[str]:
        """Decode target sequence (clean text with SOS/EOS)."""
        return self.decode_tokens(token_ids)

    def compute_cer(self, predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
        """Compute Character Error Rate."""
        import editdistance
        
        cer_scores = []
        total_edits = 0
        total_chars = 0
        
        for pred, ref in zip(predictions, references):
            if len(ref) == 0:
                cer = 1.0 if len(pred) > 0 else 0.0
            else:
                edits = editdistance.eval(pred, ref)
                cer = edits / len(ref)
            cer_scores.append(cer)
            total_edits += editdistance.eval(pred, ref)
            total_chars += len(ref)
        
        avg_cer = total_edits / total_chars if total_chars > 0 else 0.0
        return avg_cer, cer_scores

    def compute_wer(self, predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
        """Compute Word Error Rate."""
        import editdistance
        
        wer_scores = []
        total_edits = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            if len(ref_words) == 0:
                wer = 1.0 if len(pred_words) > 0 else 0.0
            else:
                edits = editdistance.eval(pred_words, ref_words)
                wer = edits / len(ref_words)
            wer_scores.append(wer)
            total_edits += editdistance.eval(pred_words, ref_words)
            total_words += len(ref_words)
        
        avg_wer = total_edits / total_words if total_words > 0 else 0.0
        return avg_wer, wer_scores

    @torch.inference_mode()
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        save_predictions: bool = False
    ) -> Dict[str, Any]:
        """Run evaluation on test set."""
        all_predictions = []
        all_references = []
        all_inputs = []
        valid_count = 0
        skipped_count = 0
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            if num_samples is not None and valid_count >= num_samples:
                break
            
            # Batch format: (distorted, clean, distorted_mask, clean_mask)
            distorted, clean, distorted_mask, clean_mask = batch
            
            distorted = distorted.to(self.device, non_blocking=True)
            distorted_mask = distorted_mask.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)
            
            batch_size = distorted.shape[0]
            
            # Generate predictions
            generated = self.generate(distorted, distorted_mask)
            
            # Decode
            pred_texts = self.decode_tokens(generated)
            ref_texts = self.decode_target(clean)
            input_texts = self.decode_tokens(distorted)
            
            # Filter valid samples
            for i in range(batch_size):
                if num_samples is not None and valid_count >= num_samples:
                    break
                if len(ref_texts[i]) == 0:
                    skipped_count += 1
                    continue
                
                all_predictions.append(pred_texts[i])
                all_references.append(ref_texts[i])
                all_inputs.append(input_texts[i])
                valid_count += 1
        
        # Compute metrics
        cer, cer_scores = self.compute_cer(all_predictions, all_references)
        wer, wer_scores = self.compute_wer(all_predictions, all_references)
        
        # Compute statistics
        ref_lengths = [len(r) for r in all_references]
        pred_lengths = [len(p) for p in all_predictions]
        avg_ref_len = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
        avg_pred_len = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
        ref_words = [len(r.split()) for r in all_references]
        pred_words = [len(p.split()) for p in all_predictions]
        avg_ref_words = sum(ref_words) / len(ref_words) if ref_words else 0
        avg_pred_words = sum(pred_words) / len(pred_words) if pred_words else 0
        
        results = {
            'cer': cer,
            'wer': wer,
            'num_samples': valid_count,
            'skipped_samples': skipped_count,
            'avg_ref_char_len': avg_ref_len,
            'avg_pred_char_len': avg_pred_len,
            'avg_ref_word_len': avg_ref_words,
            'avg_pred_word_len': avg_pred_words,
        }
        
        if save_predictions:
            results['predictions'] = [
                {
                    'input': inp,
                    'reference': ref,
                    'prediction': pred,
                    'cer': cer_scores[i] if i < len(cer_scores) else None,
                    'wer': wer_scores[i] if i < len(wer_scores) else None,
                }
                for i, (inp, ref, pred) in enumerate(zip(all_inputs, all_references, all_predictions))
            ]
        
        return results


def load_checkpoint(checkpoint_path: str, model: Module, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_state = {
        key.replace('module.', ''): value
        for key, value in state['model'].items()
    }
    model.load_state_dict(model_state)
    
    epoch = state.get('epoch', 0)
    global_step = state.get('global_step', 0)
    
    print(f"Loaded checkpoint: epoch={epoch}, global_step={global_step}")
    return {'epoch': epoch, 'global_step': global_step}


def run_evaluation(args) -> Dict[str, Any]:
    """Main evaluation function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    
    # Load tokenizer
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.special_tokens.pad_id
    
    # Create model
    model = get_model(args, 0, vocab_size, pad_idx)
    model.to(device)
    
    # Load checkpoint
    if args.checkpoint is None:
        raise ValueError("Checkpoint path must be provided via --checkpoint")
    
    load_checkpoint(args.checkpoint, model, device)
    
    # Load test data
    test_loader = get_test_loader(
        data_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_len=args.max_len,
        ratio=args.distortion_ratio,
        dist_key=args.dist_key,
        clean_key=args.clean_key,
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=getattr(args, 'pin_memory', True)
    )
    
    # Create logger
    logger = get_logger(args)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_gen_len=args.max_gen_len,
        logger=logger
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        num_samples=args.num_samples,
        save_predictions=args.save_predictions
    )
    
    # Add metadata
    results['checkpoint'] = args.checkpoint
    results['eval_batch_size'] = args.eval_batch_size
    results['max_gen_len'] = args.max_gen_len
    results['distortion_ratio'] = args.distortion_ratio
    results['dist_key'] = args.dist_key
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = Path(args.checkpoint).stem
    output_path = output_dir / f"{checkpoint_name}_metrics.json"
    save_json(output_path, results)
    print(f"Results saved to {output_path}")
    
    return results


def print_results(results: Dict[str, Any], num_examples: int = 3):
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Evaluated samples: {results['num_samples']}")
    print(f"Skipped samples: {results['skipped_samples']}")
    print(f"CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    print(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"Avg reference char length: {results['avg_ref_char_len']:.1f}")
    print(f"Avg prediction char length: {results['avg_pred_char_len']:.1f}")
    print(f"Avg reference word length: {results['avg_ref_word_len']:.1f}")
    print(f"Avg prediction word length: {results['avg_pred_word_len']:.1f}")
    
    if 'predictions' in results and results['predictions']:
        print(f"\n--- Sample Predictions (first {num_examples}) ---")
        for i, pred in enumerate(results['predictions'][:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"  INPUT (distorted):  {pred['input']}")
            print(f"  REFERENCE (clean):    {pred['reference']}")
            print(f"  PREDICTION:           {pred['prediction']}")
            print(f"  CER: {pred['cer']:.4f} | WER: {pred['wer']:.4f}")
    
    print("=" * 60)