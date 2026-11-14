#!/usr/bin/env bash
# One-time setup + start of ECHO on an Oracle Cloud Ubuntu VM (Ampere A1, ARM64).
# Run from the project folder on the VM:   bash deploy/oracle/setup.sh
# Safe to run again: every step checks whether it's already done.
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV=deploy/oracle/.env

[ -f "$ENV" ] || { echo "Missing $ENV: cp deploy/oracle/.env.example $ENV and fill in HF_TOKEN and DOMAIN"; exit 1; }
grep -q '^HF_TOKEN=hf_x' "$ENV" && { echo "Put your real HF_TOKEN in $ENV"; exit 1; }
grep -q '^DOMAIN=0-0-0-0' "$ENV" && { echo "Put your VM's DOMAIN in $ENV"; exit 1; }

echo "== 1/4 swap (4 GB), a safety net while the image builds"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== 2/4 firewall: Oracle's Ubuntu image rejects everything but SSH; open 80 and 443"
# Done before Docker is installed, so the saved rules don't include Docker's own.
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT 2>/dev/null ||
    sudo iptables -I INPUT 1 -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT
done
if command -v netfilter-persistent >/dev/null; then
  sudo netfilter-persistent save >/dev/null
else
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent >/dev/null
  sudo netfilter-persistent save >/dev/null
fi

echo "== 3/4 Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

echo "== 4/4 build and start (first build: ~20-30 min; llama.cpp is compiled for this CPU)"
sudo docker compose -f deploy/oracle/compose.yml --env-file "$ENV" up -d --build

DOMAIN=$(grep '^DOMAIN=' "$ENV" | cut -d= -f2)
echo
echo "Started. The models take ~1-2 minutes to load. Check with:"
echo "  sudo docker compose -f deploy/oracle/compose.yml logs -f echo"
echo "  curl -s https://$DOMAIN/api/echo/status"
echo "Then open https://$DOMAIN"
