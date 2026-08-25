#!/bin/bash
set -e
cd /home/user

mkdir -p /home/user/secure_config && printf "%s\n" "# HARDENED BY AUTOMATION" "PermitRootLogin no" "PasswordAuthentication no" > /home/user/secure_config/sshd_config_hardened && chmod 600 /home/user/secure_config/sshd_config_hardened && printf "%s|UPDATED|/home/user/secure_config/sshd_config_hardened\n" "$(date -u '+%F_%H:%M:%S')" >> /home/user/hardening.log
cat /home/user/secure_config/sshd_config_hardened && echo "---" && ls -l /home/user/secure_config/sshd_config_hardened && echo "---" && cat /home/user/hardening.log
