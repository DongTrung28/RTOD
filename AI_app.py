import subprocess
def check():
    subprocess.Popen(["detect.py", "--weights best.pt",""])

check()