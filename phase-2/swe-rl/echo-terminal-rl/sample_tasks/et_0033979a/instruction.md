You are an infrastructure engineer automating a quick security hardening step on a freshly-provisioned host.

INITIAL STATE  
1. A directory called /home/user/insecure_config/ already exists.  
2. Inside it there is a file named sshd_config that currently contains two (insecure) lines:  
   PermitRootLogin yes  
   PasswordAuthentication yes  

GOAL  
Create a hardened copy of this configuration and record the action in a log file.

REQUIREMENTS  
1. Make a *new* directory: /home/user/secure_config/  
2. Inside /home/user/secure_config/ create a file called sshd_config_hardened with **exactly** the three lines shown below (and nothing else).  
   Line-1: # HARDENED BY AUTOMATION  
   Line-2: PermitRootLogin no  
   Line-3: PasswordAuthentication no  
   Make sure the file ends with a single trailing newline.  
3. Ensure sshd_config_hardened has chmod 600 permissions (owner read/write only).  
4. Append one audit entry to /home/user/hardening.log in this precise pipe-delimited format:  
      YYYY-MM-DD_HH:MM:SS|UPDATED|/home/user/secure_config/sshd_config_hardened  
   • Replace “YYYY-MM-DD_HH:MM:SS” with the current UTC time when the file is created.  
   • Do not add any extra spaces.  
   • Only one line should be added for this task run.  
5. When you are finished, /home/user/secure_config/sshd_config_hardened must *exist*, contain the exact three lines shown above, have 600 permissions, and /home/user/hardening.log must contain a single correctly formatted audit line.

CONSTRAINTS  
• You do NOT need root/sudo access; stay entirely within /home/user.  
• Do not delete or rename the original /home/user/insecure_config/sshd_config file.

The automated grader will verify:  
• The presence and exact content of /home/user/secure_config/sshd_config_hardened.  
• That its numeric permissions are 600.  
• That /home/user/hardening.log exists and its only new line matches the pattern:  
      ^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\|UPDATED\|/home/user/secure_config/sshd_config_hardened$  

If all checks pass, the task is complete.