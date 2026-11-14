# ECHO on Oracle Cloud Always Free

The exact ECHO build (same models, same checks) on a free Ampere A1 VM, with HTTPS.

**What Always Free gives** (Oracle's page, checked 5 Nov 2026): Ampere A1 (ARM) up to **2 OCPUs and
12 GB RAM** in total, 200 GB of disk, in your **home region** only. ECHO needs about 4 GB while
running, so one VM with all of it is plenty.

Time: about an hour, most of it waiting for the image to build.

---

## 1. Create the account

1. Go to <https://www.oracle.com/cloud/free/> → **Start for free**.
2. **Home region**: pick the one closest to you (e.g. *India South (Hyderabad)* or *India West
   (Mumbai)*). It can't be changed later, and free VMs only run there.
3. A card is needed for identity verification (a small temporary hold, refunded). The free tier
   doesn't charge unless you upgrade *and* go over the free limits.
4. Wait for the "account is ready" email, then sign in at <https://cloud.oracle.com>.

## 2. Create the VM

Menu ☰ → **Compute** → **Instances** → **Create instance**.

| Field | Set it to |
|---|---|
| Name | `echo` |
| Image | **Change image** → *Canonical Ubuntu* → **24.04** (the aarch64 build is picked for you) |
| Shape | **Change shape** → *Ampere* → `VM.Standard.A1.Flex` → **2 OCPUs, 12 GB memory** |
| Networking | *Create new virtual cloud network* + *public subnet*; **Assign a public IPv4 address: yes** |
| SSH keys | **Generate a key pair for me** → **Save private key** (a `.key` file; keep it safe) |
| Boot volume | default (about 47-50 GB) is enough |

**Create**. In a minute or two the state is *Running*; copy the **Public IP address**.

> **"Out of capacity for shape VM.Standard.A1.Flex"** is common: free ARM machines run out in busy
> regions. Try another *Availability domain* on the same page, or retry later (early morning is easier).

## 3. Open the web ports (Oracle's side)

On the instance page: **Subnet** link → **Security Lists** → *Default Security List* →
**Add Ingress Rules**, twice:

| Source CIDR | IP protocol | Destination port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

(The VM's own firewall is opened by the setup script in step 6.)

## 4. Connect from Windows

Move the key somewhere fixed and lock it down so `ssh` accepts it (PowerShell):

```powershell
mkdir $HOME\.ssh -Force
move $HOME\Downloads\ssh-key-*.key $HOME\.ssh\oracle-echo.key
icacls $HOME\.ssh\oracle-echo.key /inheritance:r /grant:r "$($env:USERNAME):R"
ssh -i $HOME\.ssh\oracle-echo.key ubuntu@<PUBLIC_IP>
```

Answer `yes` to the fingerprint question. You're in when the prompt reads `ubuntu@echo:~$`. Type `exit` to leave.

## 5. Send the code

`D:\Projects\echo\echo-src.tar.gz` is the project without the virtual environment, models and
audio (under 1 MB; the VM downloads the models itself). From PowerShell on your PC:

```powershell
scp -i $HOME\.ssh\oracle-echo.key D:\Projects\echo\echo-src.tar.gz ubuntu@<PUBLIC_IP>:~
```

(After changing code, rebuild the archive from Git Bash: `bash /d/Projects/echo/deploy/oracle/pack.sh`.)

## 6. Configure and start (on the VM)

```bash
ssh -i ~/.ssh/oracle-echo.key ubuntu@<PUBLIC_IP>     # (from PowerShell, as in step 4)
mkdir -p ~/echo && tar xzf ~/echo-src.tar.gz -C ~/echo && cd ~/echo
cp deploy/oracle/.env.example deploy/oracle/.env
nano deploy/oracle/.env
```

In `nano`, set:

- `HF_TOKEN=` your Hugging Face **read** token (huggingface.co → Settings → Access Tokens). The same
  account must have accepted the terms on <https://huggingface.co/pyannote/speaker-diarization-3.1>
  and <https://huggingface.co/pyannote/segmentation-3.0>.
- `DOMAIN=` your IP with dashes plus `.sslip.io`, e.g. IP `140.238.12.34` → `140-238-12-34.sslip.io`.
  sslip.io is a free DNS name that points at that IP, so Caddy can get a real HTTPS certificate.

Save with `Ctrl+O`, `Enter`, exit with `Ctrl+X`. Then:

```bash
bash deploy/oracle/setup.sh
```

It adds swap, opens ports 80/443 in the VM's firewall, installs Docker, builds the image (first time
**20-30 minutes**: it compiles llama.cpp for this CPU and downloads ~3 GB of models) and starts
ECHO behind Caddy.

## 7. Check it

```bash
sudo docker compose -f deploy/oracle/compose.yml logs -f echo     # Ctrl+C to stop watching
curl -s https://<DOMAIN>/api/echo/status
```

`status` lists each model as loaded after a minute or two. Then open `https://<DOMAIN>` in a browser,
upload a recording (up to 5 minutes is processed), and ask it a question.

## Day to day

| Task | Command (in `~/echo` on the VM) |
|---|---|
| Logs | `sudo docker compose -f deploy/oracle/compose.yml logs --tail 100 echo` |
| Restart | `sudo docker compose -f deploy/oracle/compose.yml restart echo` |
| Update the code | `scp` a new archive, `tar xzf ~/echo-src.tar.gz -C ~/echo`, then `bash deploy/oracle/setup.sh` |
| Stop everything | `sudo docker compose -f deploy/oracle/compose.yml down` |

The containers restart on their own after a reboot (`restart: unless-stopped`).

## Things to know

- **Idle reclaim.** Oracle may stop an Always Free VM that is idle for 7 days: CPU (95th percentile),
  network *and* memory all under 20%. ECHO keeps its models in memory (~4 GB of 12 GB, over 20%), so it
  shouldn't count as idle. Upgrading the account to *Pay As You Go* removes the rule entirely; free
  resources stay free, and a budget alert (Billing → Budgets, $1) warns you if anything ever costs money.
- **Speed.** 2 ARM cores: expect roughly real time (5 minutes of audio in about 5-7 minutes). One
  recording is processed at a time; others queue.
- **Security.** Only ports 22, 80 and 443 are open. Uploads are limited per visitor per hour by the
  app. Keep the `.key` file and `.env` private.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `ssh: Permissions for ... are too open` | rerun the `icacls` line in step 4 |
| `ssh` times out | the instance isn't *Running*, or you used the private instead of the public IP |
| Build stops with `Killed` | out of memory while compiling: check `free -h` shows the 4 GB swap, rerun `setup.sh` |
| Browser can't reach the site | step 3 rules missing; `sudo iptables -L INPUT -n --line-numbers` should show ACCEPT for 80 and 443 above the REJECT line |
| Caddy log: certificate errors | ports 80/443 must be reachable from the internet first; `DOMAIN` must match the IP exactly |
| status: diarizer error mentioning the token | accept both pyannote model terms with the token's account; fix `HF_TOKEN` in `.env`; `sudo docker compose -f deploy/oracle/compose.yml up -d` |
| `illegal instruction` in the echo log | send me the log; llama.cpp's build flags need adjusting for the CPU |
