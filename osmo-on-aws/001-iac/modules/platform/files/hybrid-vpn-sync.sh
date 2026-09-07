#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Reconcile this host with the WireGuard gateway configuration Terraform
# published to Parameter Store.
#
# Idempotent, and run from a systemd timer rather than once at boot. That is what
# makes a boot which raced the elastic IP association, or hit a transient package
# mirror failure, recover on its own — and what applies a peer list edited after
# the instance was created without replacing the instance.
set -euo pipefail

# shellcheck source=/dev/null
. /etc/default/hybrid-vpn
export AWS_DEFAULT_REGION="$AWS_REGION"

if ! command -v wg >/dev/null || ! command -v aws >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq wireguard-tools awscli
fi

# The instance routes for addresses that are not its own, so this and the ENI's
# disabled source/destination check are both mandatory.
if [ "$(sysctl -n net.ipv4.ip_forward)" != "1" ]; then
  echo 'net.ipv4.ip_forward = 1' > /etc/sysctl.d/99-hybrid-vpn.conf
  sysctl -q -w net.ipv4.ip_forward=1
fi

# The server key is generated here and kept in Parameter Store, so no private key
# reaches the Terraform state, the user data, or the repository. Generated once:
# regenerating it would invalidate every client configuration already handed out.
priv=$(aws ssm get-parameter --name "$KEY_PARAM" --with-decryption \
  --query Parameter.Value --output text 2>/dev/null || true)
if [ -z "$priv" ]; then
  priv=$(wg genkey)
  aws ssm put-parameter --name "$KEY_PARAM" --type SecureString \
    --key-id "$KMS_KEY_ID" --value "$priv" --overwrite > /dev/null
fi

# Only written when it changed: Parameter Store keeps a bounded number of
# versions per parameter, and this runs every minute.
pub=$(printf '%s' "$priv" | wg pubkey)
if [ "$(aws ssm get-parameter --name "$PUB_PARAM" --query Parameter.Value \
  --output text 2> /dev/null || true)" != "$pub" ]; then
  aws ssm put-parameter --name "$PUB_PARAM" --type String --value "$pub" --overwrite > /dev/null
fi

peers=$(aws ssm get-parameter --name "$PEERS_PARAM" --query Parameter.Value --output text)

mkdir -p /etc/wireguard
umask 077
rendered=$(mktemp)
{
  echo '[Interface]'
  echo "Address = $TUNNEL_ADDRESS"
  echo "ListenPort = $LISTEN_PORT"
  echo "PrivateKey = $priv"
  echo
  printf '%s\n' "$peers"
} > "$rendered"

# A full restart rather than `wg syncconf`: syncconf updates peers but not the
# routes wg-quick derives from their AllowedIPs, so a peer added later would have
# no route and be silently unreachable.
if ! cmp -s "$rendered" /etc/wireguard/wg0.conf; then
  install -m 600 "$rendered" /etc/wireguard/wg0.conf
  systemctl restart wg-quick@wg0
elif ! systemctl is-active --quiet wg-quick@wg0; then
  systemctl enable --now wg-quick@wg0
fi
rm -f "$rendered"
