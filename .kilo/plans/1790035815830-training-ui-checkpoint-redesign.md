# Training Console UI Redesign & Automated Checkpointing Plan

## Goal
Redesign the training console UI for real-time monitoring clarity and implement automated step-based checkpointing (every 5,000 steps) with global rotation of 5 checkpoints.

---

## Current State Analysis

| Component | Current Implementation |
|-----------|------------------------|
| **Progress Bar** | `tqdm` on train/test loaders |
| **Console Logging** | `BasicLogger.log_step()` prints per-step loss; `log()` prints per-epoch |
| **TensorBoard** | `TBLogger` extends `BasicLogger` with SummaryWriter |
| **Checkpointing** | `save_ckpt(epoch)` → `checkpoint_{epoch}.pt` triggered by `TerminationCallback` on validation improvement |
| **Rotation** | None - all checkpoints persist |

---

## Design Decisions (from user)

| Decision | Choice |
|----------|--------|
| **Console Metrics** | Train loss, Learning Rate, Step Time, Epoch ETA |
| **Checkpoint Policy** | Supplement: keep epoch-based on improvement + add step-based every 5K steps |
| **UI Update Style** | Overwrite in-place (single line, like tqdm) |
| **Rotation Scope** | Global pool of 5 (all checkpoint types share one rotation) |

---

## Implementation Plan

### 1. Console UI Redesign (`core/train.py`)

#### 1.1 Create `TrainingMonitor` class
```python
class TrainingMonitor:
    def __init__(self, total_steps_per_epoch: int, log_interval: int = 100):
        self.total_steps = total_steps_per_epoch
        self.log_interval = log_interval
        self.epoch_start_time = None
        self.step_times = []
        self.current_epoch = 0
        self.global_step = 0
    
    def on_epoch_start(self, epoch: int):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
    
    def on_step_end(self, step: int, loss: float, lr: float):
        self.global_step = step
        now = time.time()
        self.step_times.append(now)
        if len(self.step_times) > 100:  # rolling window for ETA
            self.step_times.pop(0)
        
        if step % self.log_interval == 0:
            self._render_line(loss, lr)
    
    def _render_line(self, loss: float, lr: float):
        # Calculate step time (avg of recent)
        # Calculate ETA: remaining_steps * avg_step_time
        # Format: [Epoch 3/10] Step 15000/27040 | Loss: 1.234 | LR: 2.1e-4 | Step: 0.42s | ETA: 12m34s
        # Use \r for in-place overwrite
        pass
```

#### 1.2 Integrate into `DistTrainer.train()`
- Replace `tqdm` with custom loop + `TrainingMonitor`
- Call `monitor.on_step_end()` each accumulation step
- Remove per-step `logger.log_step()` calls (monitor handles display)
- Keep TensorBoard logging via existing `logger.log_step()`

#### 1.3 Validation Progress
- Keep `tqdm` for validation (shorter, epoch-level is fine)
- Or use same monitor for consistency

---

### 2. Automated Step-Based Checkpointing (`core/train.py`)

#### 2.1 Add Checkpoint Manager
```python
class CheckpointManager:
    def __init__(self, outdir: Path, max_keep: int = 5):
        self.outdir = Path(outdir)
        self.max_keep = max_keep
        self.checkpoints = []  # list of (step, epoch, path, type)
    
    def save(self, trainer, step: int, epoch: int, ckpt_type: str):
        # ckpt_type: "step" or "epoch"
        path = self.outdir / f"checkpoint_step{step}_epoch{epoch}_{ckpt_type}.pt"
        trainer.save_ckpt_to_path(path)  # new method
        self.checkpoints.append((step, epoch, path, ckpt_type))
        self._rotate()
    
    def _rotate(self):
        while len(self.checkpoints) > self.max_keep:
            oldest = self.checkpoints.pop(0)
            if oldest[2].exists():
                oldest[2].unlink()
```

#### 2.2 Add Step Counter to `DistTrainer`
- Track `self.global_step` (incremented each optimizer step)
- Every 5,000 steps: call `checkpoint_manager.save(trainer, step, epoch, "step")`

#### 2.3 Modify Existing `save_ckpt()`
- Add `save_ckpt_to_path(path)` for custom paths
- Keep `save_ckpt(epoch)` for epoch-based (called by callback)
- Both go through `CheckpointManager`

#### 2.4 Update `TerminationCallback` Interaction
- Callback still triggers epoch-based saves on improvement
- Both checkpoint types enter same rotation pool

---

### 3. Configuration (`core/args.py`)

Add new args:
```python
# Checkpointing
parser.add_argument('--ckpt_interval_steps', default=5000, type=int,
                    help='Save checkpoint every N steps')
parser.add_argument('--max_checkpoints', default=5, type=int,
                    help='Maximum checkpoints to keep (rotation)')
```

---

### 4. Files to Modify

| File | Changes |
|------|---------|
| `core/train.py` | Add `TrainingMonitor`, `CheckpointManager`, integrate into `DistTrainer` |
| `core/args.py` | Add `--ckpt_interval_steps`, `--max_checkpoints` |
| `core/callback.py` | No changes (callback still works, checkpoints go through manager) |

---

## Detailed Task List

### Phase 1: Console UI
- [ ] Create `TrainingMonitor` class in `core/train.py`
- [ ] Replace `tqdm` loop in `train()` with custom loop + monitor
- [ ] Implement in-place line rendering with `\r` and ANSI clear
- [ ] Display: Epoch, Step/Total, Loss, LR, Step Time, ETA
- [ ] Keep TensorBoard logging unchanged

### Phase 2: Checkpointing System
- [ ] Create `CheckpointManager` class
- [ ] Add `global_step` counter to `DistTrainer`
- [ ] Add `save_ckpt_to_path(path)` method
- [ ] Trigger step-based save every `ckpt_interval_steps` (default 5000)
- [ ] Implement rotation logic (keep latest 5 globally)
- [ ] Ensure epoch-based saves from callback also use manager

### Phase 3: Configuration & Integration
- [ ] Add CLI args for checkpoint interval and max keep
- [ ] Wire everything in `get_trainer()` / `DistTrainer.__init__`
- [ ] Test: run training, verify step checkpoints appear, rotation works

---

## Edge Cases & Validation

| Scenario | Handling |
|----------|----------|
| Resume from checkpoint | Load `global_step` from optimizer state; continue counting |
| Step interval < epoch length | Multiple step checkpoints per epoch (expected) |
| Step interval > epoch length | May skip some epochs for step saves (callback handles epoch saves) |
| Training interrupted | Latest 5 checkpoints always available |
| Disk full | `torch.save` raises; catch and log warning, don't crash |

---

## Testing Checklist

- [ ] Start training, verify single-line console updates
- [ ] Verify metrics: Loss, LR, Step Time, ETA display correctly
- [ ] At step 5000: `checkpoint_step5000_epochX_step.pt` created
- [ ] At step 10000: second step checkpoint created
- [ ] After 6 checkpoints: oldest deleted automatically
- [ ] Epoch-based improvement checkpoints also enter rotation
- [ ] Resume from step checkpoint works (loads model, optimizer, step counter)
- [ ] TensorBoard still receives all metrics

---

## Out of Scope

- Multi-GPU / distributed training (project is single-GPU only)
- Rich/TUI library dependency (keep stdlib only)
- Checkpoint compression or sharding
- Web-based dashboard

---

## Plan File
`.kilo/plans/1790035815830-training-ui-checkpoint-redesign.md`