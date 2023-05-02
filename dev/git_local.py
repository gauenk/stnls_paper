"""

git clone all locally required packages

"""

import subprocess

def main():
    pkgs = ["dev_basics","cache_io","dev_basics","data_hub",
            "n3net","colanet","lidia","mvit","frame2frame"]
    cmd_fmt = "cd ../%s; git pull; cd ../stnls_paper"
    check_me = []
    check_me_msg = []
    for pkg in pkgs:
        cmd = (cmd_fmt % pkg)#.split(" ")
        proc = subprocess.run(cmd, shell=True, capture_output=True)
        stdout = proc.stdout
        stderr = proc.stderr
        if len(stderr) > 0:
            check_me.append(pkg)
            check_me_msg.append(stderr)
    print(check_me)
    for i in range(len(check_me)):
        print(check_me[i],check_me_msg[i])
    

if __name__ == "__main__":
    main()
