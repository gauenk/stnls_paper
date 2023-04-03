#!/home/gauenk/.pyenv/shims/python

import subprocess
pkgs = ["dev_basics","cache_io","dev_basics","data_hub",
        "n3net","colanet","lidia","mvit"]
cmd_fmt = "cd ../%s; git pull; cd ../stnls_paper"
check_me = []
for pkg in pkgs:
    cmd = (cmd_fmt % pkg)#.split(" ")
    proc = subprocess.run(cmd, shell=True, capture_output=True)
    stdout = proc.stdout
    stderr = proc.stderr
    if len(stderr) > 0:
        check_me.append(pkg)
print(check_me)
