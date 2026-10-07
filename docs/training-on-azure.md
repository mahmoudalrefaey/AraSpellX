# Training on Azure

This guide runs AraSpellX pretraining and correction training on an Azure GPU virtual machine. The data is built locally (see the README); only the finished training files are uploaded.

Commands marked **local** run in PowerShell on your own computer. Commands marked **VM** run in the virtual machine's Linux shell.

---

## 1. Before you start: subscription, GPU quota and costs

**Subscription.** GPU machines need a pay-as-you-go subscription (credits on it are fine).
- Free-trial accounts and *Azure for Students* usually have a GPU quota of 0, and requests to raise it are usually refused.
- If yours is a free trial, upgrade it to pay-as-you-go in the portal. Your remaining credit stays.

**Choosing a GPU.**

| VM size | GPU | Precision | Notes |
|---|---|---|---|
| `Standard_NC4as_T4_v3` | 1× T4, 16 GB | fp16 | Cheapest. Similar to or a bit slower than an RTX 3060 laptop GPU. Quota family **NCASv3_T4** (4 vCPUs) |
| `Standard_NC24ads_A100_v4` | 1× A100, 80 GB | bf16 | Several times faster. Quota family **NCADS_A100_v4** (24 vCPUs) |

The older V100 sizes (NCv3) have been retired, so don't use them.

**Cost, approximately.** Prices vary by region and change over time; check the [pricing page](https://azure.microsoft.com/pricing/details/virtual-machines/linux/) before you start.

| | T4 | A100 |
|---|---|---|
| Pay-as-you-go | about $0.5/hour | about $3.7/hour |
| Spot | often 60–80% cheaper | often 60–80% cheaper |

- **Spot VMs** can be evicted (stopped) at any time. Training checkpoints every 20 minutes and resumes from the last checkpoint, so an eviction costs at most about 20 minutes of work.
- **You pay for a VM until it is deallocated**, even if you shut the OS down from inside the VM. Always stop it from the portal or with `az vm deallocate` (section 8).
- **Disks keep costing** a few dollars a month until you delete them.

**Set a budget alert first:** Portal → *Cost Management* → *Budgets* → *Add*, with an email alert at 50% and 90%.

## 2. Install the Azure CLI and log in (local)

```powershell
winget install -e --id Microsoft.AzureCLI
# open a new PowerShell window, then:
az login
az account show --output table            # confirm the right subscription
# az account set --subscription "<name or id>"   # if you have several
```

## 3. Request GPU quota

**Pick a region.** It must offer the size you want:

```powershell
az vm list-skus --location westeurope --size Standard_NC4as_T4_v3 --output table
az vm list-skus --location westeurope --size Standard_NC24ads_A100_v4 --output table
```

An empty table, or a `Restrictions` column showing `NotAvailableForSubscription`, means that region won't work. Try `eastus`, `northeurope`, `southcentralus` or `westus2`.

**See your current quota:**

```powershell
az vm list-usage --location westeurope --output table | Select-String "T4|A100|Spot"
```

**Request more**:
1. Go to Portal → *Quotas* → *Compute*.
2. Filter by your region.
3. Find the family:
   - **Standard NCASv3_T4 Family vCPUs**: request 4.
   - **Standard NCADS_A100_v4 Family vCPUs**: request 24.
4. Click the pencil icon and submit.

For spot VMs, also request **Total Regional Spot vCPUs** (4 for T4, 24 for A100).

Small requests are often approved within minutes; others go to a support ticket that can take a day or two.

## 4. Create the VM (local)

This uses the **Data Science Virtual Machine** image (Ubuntu 22.04), which comes with NVIDIA drivers installed.

```powershell
$RG = "araspellx-rg"
$VM = "araspellx-gpu"
$LOC = "westeurope"
$SIZE = "Standard_NC4as_T4_v3"        # or Standard_NC24ads_A100_v4

az group create --name $RG --location $LOC

az vm create `
  --resource-group $RG --name $VM --location $LOC `
  --size $SIZE `
  --image microsoft-dsvm:ubuntu-2204:2204-gen2:latest `
  --security-type Standard `
  --os-disk-size-gb 128 `
  --admin-username azureuser --generate-ssh-keys `
  --public-ip-sku Standard
```

**For a spot VM**, add these flags. `-1` means you pay up to the normal price and won't be evicted for price reasons:

```powershell
  --priority Spot --eviction-policy Deallocate --max-price -1 `
```

**If the image can't be found**, list the current ones:

```powershell
az vm image list --publisher microsoft-dsvm --offer ubuntu-2204 --all --output table
```

Or use plain Ubuntu (`--image Ubuntu2204`) and install the driver yourself in section 5.

**Note the public IP address** from the output, or get it later:

```powershell
$IP = az vm show -d -g $RG -n $VM --query publicIps -o tsv
```

**Safety net: automatic deallocation.** This deallocates the VM every day at the given UTC time. Set it a few hours after you expect training to finish; if training is still running, rerun the command after restarting the VM and it resumes.

```powershell
az vm auto-shutdown -g $RG -n $VM --time 0600
```

## 5. Set up the VM

Connect:

```powershell
ssh azureuser@$IP
```

**VM**: check the GPU:

```bash
nvidia-smi        # must show the T4 or A100
```

If `nvidia-smi` is missing (plain Ubuntu image), install the driver and reconnect:

```bash
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install && sudo reboot
```

**VM**: install the project:

```bash
sudo apt update && sudo apt install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone -b v1 https://github.com/mahmoudalrefaey/AraSpellX.git
cd AraSpellX
uv sync                        # installs Python 3.10 and PyTorch with CUDA
uv run python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
mkdir -p data/v1/pretrain data/v1/pairs data/v1/testsets artifacts
```

## 6. Upload the data (local)

Run these from the project folder on your computer.

| Stage | Files | Size |
|---|---|---|
| Pretraining | `data/v1/pretrain/train.bin`, `valid.bin` | 2.0 GB |
| Correction | the files above, plus `data/v1/pairs/*.jsonl` and `data/v1/testsets/yarmouk_dev.jsonl` | ~80 MB more |
| Correction, if pretraining ran elsewhere | also `artifacts/pretrain/model/` | ~60 MB |

```powershell
cd D:\AraSpellX
scp data/v1/pretrain/train.bin data/v1/pretrain/valid.bin data/v1/pretrain/stats.json azureuser@${IP}:~/AraSpellX/data/v1/pretrain/
scp data/v1/pairs/*.jsonl azureuser@${IP}:~/AraSpellX/data/v1/pairs/
scp data/v1/testsets/yarmouk_dev.jsonl azureuser@${IP}:~/AraSpellX/data/v1/testsets/
# only if the pretrained model comes from another machine:
scp -r artifacts/pretrain/model azureuser@${IP}:~/AraSpellX/artifacts/pretrain/
```

`scp` can't resume an interrupted transfer; if one fails, run it again.

On the VM, check the files arrived whole. The sizes must match the local ones:

```bash
ls -l data/v1/pretrain
```

## 7. Train

**Always run training inside `tmux`**, so it keeps running when you disconnect:

| Action | Command |
|---|---|
| Start a session | `tmux new -s train` |
| Leave it running (detach) | `Ctrl+b`, then `d` |
| Come back later | `tmux attach -t train` |

### Precision

`--precision auto` (the default) picks the right setting, so you normally don't need to pass anything:
- **bf16** on A100 (and on the RTX 30xx laptop)
- **fp16** with loss scaling on T4

The first log line shows which one was chosen.

### Pretraining

1. **Check run** (a few minutes). Verify that:
   - the loss falls
   - `acc` rises
   - the two evaluations print validation numbers and example fills

   ```bash
   uv run python -m araspellx.train.pretrain --out artifacts/pretrain_check --max_steps 600 --eval_every 200
   ```

2. **Full run**:

   ```bash
   uv run python -m araspellx.train.pretrain --out artifacts/pretrain --max_steps 40000
   ```

**Estimate the duration** from the check run. The bar shows steps per second, so the full run takes about `40000 / (steps per second) / 3600` hours. The bar's ETA becomes reliable after a few hundred steps.

### Correction training

Run this after pretraining has finished. It reads the encoder from `artifacts/pretrain/model`.

```bash
uv run python -m araspellx.train.correct --out artifacts/correct_check --max_steps 600 --eval_every 300
uv run python -m araspellx.train.correct --out artifacts/correct --max_steps 30000
```

### Batch size

Keep the default batch sizes (64 for pretraining, 48 for correction) so results are comparable with laptop runs; both fit on a T4. Changing the batch size changes the training recipe, so if you do, scale `--lr` with it.

### Resuming

If the VM is evicted, auto-shut down, or training stops for any reason:
1. Start the VM: `az vm start -g $RG -n $VM`
2. Connect and open tmux.
3. Run **the same command** again. It continues from `last.pt` (at most 20 minutes of work is lost).

The public IP may change after a restart; get it again with the `az vm show` command from section 4.

## Monitoring

**The progress bar** (`tmux attach -t train`) shows:

| Field | Meaning |
|---|---|
| step / total, speed, ETA | how far along the run is |
| `loss` | training loss (smoothed) |
| `acc` (pretraining) | how often a masked character is predicted correctly |
| `edit acc` (correction) | how often a character that needs an edit gets the right edit |
| `lr` | the current learning rate |
| `char/s` or `win/s` | training throughput |
| `GPU GB` | peak GPU memory |

Evaluations and checkpoints are printed above the bar with timestamps.

**The log file** holds every message plus a metrics line every 100 steps. You can read it without attaching:

```bash
tail -f artifacts/pretrain/train.log           # live; Ctrl+C to stop following
cat artifacts/pretrain/eval.jsonl              # one line per evaluation
```

**GPU load:**

```bash
watch -n 5 nvidia-smi          # GPU utilization should stay high (above ~80%)
```

**TensorBoard charts** (loss, accuracy, learning rate, and evaluation precision/recall/F0.5/damage):

```bash
# VM, in a second tmux window (Ctrl+b, then c):
uv run tensorboard --logdir artifacts --port 6006
```

```powershell
# local: tunnel the port, then open http://localhost:6006
ssh -N -L 6006:localhost:6006 azureuser@$IP
```

### What healthy runs look like

**Pretraining:**
- Loss falls quickly in the first few thousand steps, then slowly.
- Masked-character accuracy rises steadily.
- The example fills at each evaluation turn into real Arabic words, often the hidden ones.
- `loss nan`, or a loss that is flat from the start, means stop and investigate.

**Correction:**
- The `clean` set's damage should stay near 0.
- Precision on `typed`, `ocr_render` and `yarmouk_real` should rise above recall.
- `cer_out` should fall below `cer_in`.

## 8. Download results and stop paying (local)

**Download** the model and logs. The checkpoint `last.pt` is only needed to continue training.

```powershell
scp -r azureuser@${IP}:~/AraSpellX/artifacts/pretrain/model artifacts/pretrain/
scp azureuser@${IP}:~/AraSpellX/artifacts/pretrain/train.log azureuser@${IP}:~/AraSpellX/artifacts/pretrain/eval.jsonl artifacts/pretrain/
# correction stage:
scp -r azureuser@${IP}:~/AraSpellX/artifacts/correct/model artifacts/correct/
```

**Stop paying for compute** (the disk and data are kept):

```powershell
az vm deallocate -g $RG -n $VM
az vm show -d -g $RG -n $VM --query powerState -o tsv   # must say "VM deallocated"
```

**When you are completely done**, delete everything. This deletes the VM, disk, IP and data on Azure, and can't be undone:

```powershell
az group delete --name $RG --yes
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Operation could not be completed as it results in exceeding approved ... quota` | Request quota for that family and region (section 3), or try another region |
| `SkuNotAvailable` / `AllocationFailed` | The region has no free GPUs of that size right now. Try another region, or a non-spot VM |
| `nvidia-smi: command not found` | Install the driver (section 5) |
| `torch.cuda.is_available()` is False | Run `nvidia-smi`; if it works, run `uv sync` again inside `~/AraSpellX` |
| `CUDA out of memory` | Lower `--batch_size` (and `--lr` in proportion), or use a larger GPU |
| Training stopped when SSH disconnected | It wasn't running inside tmux; rerun the same command inside tmux and it resumes |
| `loss nan` with fp16 | Restart from the last checkpoint with `--precision fp32` (slower), and report it |
