FROM ubuntu:22.04

# Install Python 3, pip and pytest
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip && \
    pip3 install --no-cache-dir pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Prepare initial directory structure and files
RUN mkdir -p /home/user/insecure_config && \
    chmod 0777 /home/user

# Create the insecure sshd_config with the required two lines
RUN cat >/home/user/insecure_config/sshd_config <<'EOF'
PermitRootLogin yes
PasswordAuthentication yes
EOF

RUN chmod 0644 /home/user/insecure_config/sshd_config && \
    chmod 0755 /home/user/insecure_config

LABEL Author="hardening-task-builder"
LABEL Description="Environment for security hardening automation task. Includes pytest and the initial insecure sshd_config setup."

# Help:
# This container provides the starting point for the security-hardening automation
# exercise.  It contains:
#   • /home/user/insecure_config/sshd_config with two insecure directives.
#   • Python 3 and pytest for automated verification.