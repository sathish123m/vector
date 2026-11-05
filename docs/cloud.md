# Cloud setup — Oracle Cloud Always Free (ARM)

The project is CPU-only Python/NumPy. A 24 GB ARM VM with a 200 GB persistent
disk is enough to run the remaining SIFT100K churn and SIFT1M experiments and
to host the future API.

This guide assumes you have an Oracle Cloud account. Account creation needs a
credit card for identity verification; stay on **Always Free** resources so
you are not billed.

## 1. Create the VM (Oracle console)

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

## 2. Open the SSH port

The default security list usually allows TCP 22. Confirm under
**Networking → VCN → Security Lists → Default Security List**. Later, for the
API demo, add an ingress rule for TCP 8000 from your IP only.

## 3. One-command bootstrap

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

The script:

1. Installs `git`, `make`, `build-essential`, `python3-dev`, `python3-venv`,
   `curl`.
2. Creates an 8 GB swap file (safety margin for SIFT1M).
3. Installs `uv`.
4. Clones/updates the repository at `~/churnvec`.
5. Runs `make setup`, `make data`, `make test`, and `make e0`.
6. Prints Python, NumPy, memory, and disk information.

It is idempotent: re-running it updates the clone and re-checks the
environment.

## 4. Run the benchmarks

```bash
cd "$HOME/churnvec"

make e1          # SIFT100K static gate
make e2          # SIFT10K churn + compaction
make e2-100k     # SIFT100K churn (24 GB host)
make e3          # SIFT1M static (shipped GT, 10K queries)
```

Or run them sequentially with logs:

```bash
bash scripts/cloud/run_cloud_benchmarks.sh
```

Optional flags:

```bash
RUN_100K=1 bash scripts/cloud/run_cloud_benchmarks.sh
RUN_1M=1   bash scripts/cloud/run_cloud_benchmarks.sh
```

Logs are written under `logs/cloud/` (gitignored).

## 5. API demo (later)

```bash
# once the API exists
.venv/bin/uvicorn churnvec.api:app --host 0.0.0.0 --port 8000
```

For a panel demo, prefer a temporary tunnel (Cloudflare Tunnel or ngrok) over
leaving port 8000 open. If you do expose it, restrict the security-list rule
to your own IP and add a token.

## Benchmark-host policy

The Week 1–14 numbers in the README were measured on the reference Apple M1
laptop. Cloud numbers are a **different host**. Do not mix them.

Either:

- keep the M1 as the reference host and label cloud results as validation, or
- re-run E0/E1/E2 on the cloud VM and report that host consistently.

Record the CPU model, core count, RAM, and OS in the result JSON (the harness
already captures platform, machine, CPU count, and total RAM).

## Free-tier caveats

- Credit card required at signup; stay on Always Free resources.
- ARM capacity varies by region; retries may be needed.
- Keep the instance active; free-tier resources can be reclaimed if unused
  for long periods.
- The 200 GB block-storage allowance is shared across the tenancy.
- Back up code to GitHub and results to your laptop or object storage.
