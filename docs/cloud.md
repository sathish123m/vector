# Cloud setup

The project is CPU-only Python/NumPy. The remaining heavy runs are SIFT100K
churn and SIFT1M; they need RAM headroom for exact ground truth but no GPU.

## Choose a host

| Host | Card needed | RAM | Session | Best for |
|---|---|---:|---|---|
| **Kaggle Notebook** | **No** | ~30 GB | 12 h | SIFT100K churn and SIFT1M |
| **GitHub Codespaces** | **No** | 16 GB | 120 core-hours/month | Editing, tests, SIFT100K churn |
| **Azure for Students** | **No** (academic email) | 16 GB+ | credit-limited | Heavy runs and API demo |
| Oracle Cloud Always Free | Visa/Mastercard/Amex | 24 GB | 24/7 | Long-lived VM and API hosting |
| Google Cloud free trial | Credit card | configurable | 90 days | A single heavy run |

Oracle does **not** accept Discover-family cards, so it is only an option if a
Visa, Mastercard, or Amex is available.

---

## Option A — Kaggle (no card, recommended now)

Kaggle provides a free CPU notebook with about 30 GB RAM, a 20 GB working
directory, and sessions of up to 12 hours. It requires no payment method.

1. Create an account at <https://www.kaggle.com>.
2. Enable **Internet** in the notebook settings. Kaggle asks for one-time
   phone verification the first time.
3. Create a notebook and import
   [`notebooks/kaggle_benchmarks.ipynb`](../notebooks/kaggle_benchmarks.ipynb),
   or paste these two cells:

   ```python
   !git clone https://github.com/sathish123m/vector.git /kaggle/working/churnvec 2>/dev/null || git -C /kaggle/working/churnvec pull --ff-only
   ```

   ```python
   !TARGETS="e2-100k e3" bash /kaggle/working/churnvec/scripts/cloud/kaggle_setup.sh
   ```

4. Set **Accelerator = None** and use **Save Version → Save & Run All
   (Commit)** so the run continues after you close the browser.
5. When it finishes, download `results/` or `churnvec_results.zip` from the
   notebook **Output** panel.

`scripts/cloud/kaggle_setup.sh`:

- updates the clone and installs the package with `pip`;
- finds SIFT in this order: extracted `.fvecs` files from an attached Kaggle
  Dataset, the official archives in a Kaggle Dataset, the official TEXMEX FTP
  host, then the ann-benchmarks HTTPS HDF5 mirror
  (`sift-128-euclidean.hdf5`), converting it to the harness layout;
- runs the test suite, then the requested `make` targets;
- archives `results/` and `logs/` after **every** target, so a late failure
  does not lose earlier evidence.

Fastest data option: use **Add Data** in the notebook to attach a public
Kaggle dataset that already contains `sift_base.fvecs` (search for `sift1m`).
The script detects it automatically. The HDF5 mirror holds the same SIFT1M
vectors in a different container; it is not byte-identical to the TEXMEX
archive, so label the data source in any report that uses it.

Run the full cloud sweep instead of just the heavy pair:

```bash
TARGETS="e1 e2 e2-100k e3" bash scripts/cloud/kaggle_setup.sh
```

Use the same host for every number you intend to compare. Kaggle sessions are
not a 24/7 service, so they are for benchmarks, not for hosting the API.

---

## Option B — GitHub Codespaces (no card, full terminal)

Codespaces gives a full Ubuntu terminal with 16 GB RAM, 32 GB disk, and 120
core-hours per month on the free plan. It needs no card, and it is the closest
no-card equivalent of the Oracle VM.

1. On GitHub, open the repository → **Code → Codespaces → Create codespace**.
2. In the terminal:

   ```bash
   bash scripts/cloud/setup_oracle_vm.sh
   make e2-100k
   ```

The Oracle bootstrap script is distro-agnostic Ubuntu setup and works here.
16 GB is enough for SIFT100K churn; use Kaggle for SIFT1M if memory gets tight.
Codespaces stops after idle timeout and is not suitable for a permanent public
API.

---

## Option C — student credits (no card, if you have a college email)

Microsoft **Azure for Students** gives a credit without a credit card, verified
through an academic email address. If your institution qualifies, create an
Ubuntu 24.04 VM (for example `Standard_D4s_v5`, 4 vCPU / 16 GB) and run the
same `scripts/cloud/setup_oracle_vm.sh` bootstrap, then `make e2-100k` and
`make e3`. The credit is time-limited, so use it for the heavy runs and the
API demo, not as a permanent host.

---

## Option D — Oracle Cloud Always Free (card required)

A 24 GB ARM VM with a 200 GB persistent disk can run every experiment and host
the API 24/7. Oracle accepts **Visa, Mastercard, and Amex only**.

### 1. Create the VM (Oracle console)

1. Sign in to <https://cloud.oracle.com>.
2. **Compute → Instances → Create instance**.
3. Name: `churnvec`.
4. Image: **Canonical Ubuntu 24.04** (aarch64).
5. Shape: **VM.Standard.A1.Flex**, **4 OCPUs**, **24 GB memory**.
6. Boot volume: **100–200 GB**.
7. Networking: create/select a VCN with a **public subnet** and assign a
   **public IPv4 address**.
8. SSH keys: upload your public key or let Oracle generate a key pair.
9. Create the instance and wait for **Running**. Note the public IP.

If the console says the shape is out of capacity, retry later or choose
another availability domain/region. ARM capacity is not guaranteed.

### 2. Open the SSH port

The default security list usually allows TCP 22. Confirm under
**Networking → VCN → Security Lists → Default Security List**. Later, for the
API demo, add an ingress rule for TCP 8000 from your IP only.

### 3. One-command bootstrap

SSH into the VM:

```bash
ssh -i /path/to/private-key ubuntu@<PUBLIC_IP>
```

Then run:

```bash
curl -fsSL https://raw.githubusercontent.com/sathish123m/vector/main/scripts/cloud/setup_oracle_vm.sh -o setup.sh
bash setup.sh
```

Or clone first and run the script from the repo:

```bash
git clone https://github.com/sathish123m/vector.git "$HOME/churnvec"
cd "$HOME/churnvec"
bash scripts/cloud/setup_oracle_vm.sh
```

The script installs system packages, creates an 8 GB swap file, installs
`uv`, clones/updates the repository at `~/churnvec`, and runs `make setup`,
`make data`, `make test`, and `make e0`. It is idempotent: re-running it
updates the clone and re-checks the environment.

### 4. Run the benchmarks

```bash
cd "$HOME/churnvec"

make e1          # SIFT100K static gate
make e2          # SIFT10K churn + compaction
make e2-100k     # SIFT100K churn
make e3          # SIFT1M static (shipped GT, 10K queries)
```

Or run them sequentially with logs:

```bash
RUN_100K=1 RUN_1M=1 bash scripts/cloud/run_cloud_benchmarks.sh
```

Logs are written under `logs/cloud/` (gitignored).

### 5. API demo (later)

```bash
# once the API exists
.venv/bin/uvicorn churnvec.api:app --host 0.0.0.0 --port 8000
```

For a panel demo, prefer a temporary tunnel (Cloudflare Tunnel or ngrok) over
leaving port 8000 open. If you do expose it, restrict the security-list rule
to your own IP and add a token.

---

## Benchmark-host policy

The Week 1–14 numbers in the README were measured on the reference Apple M1
laptop. Cloud numbers are a **different host**. Do not mix them.

Either:

- keep the M1 as the reference host and label cloud results as validation, or
- re-run E0/E1/E2 on the cloud host and report that host consistently.

Record the CPU model, core count, RAM, and OS in the result JSON (the harness
already captures platform, machine, CPU count, and total RAM).

## Caveats

- **Kaggle:** 12-hour sessions and a 20 GB working directory. Enable Internet
  (phone verification). Download results before the session ends.
- **Codespaces:** stops after an idle timeout; 120 core-hours per month on the
  free plan. Not a permanent server.
- **Oracle:** card required; ARM capacity varies by region; keep the instance
  active because free-tier resources can be reclaimed if unused for long
  periods. The 200 GB block-storage allowance is shared across the tenancy.
- In every case: code lives on GitHub, data is re-downloadable with
  `make data`, and results should be copied back to the laptop.
