#!/bin/bash
# import multiple remote git repositories to local CODE dir

# settings / change this to your config
remoteHost=github
remoteUser=gauenk
remoteDir="~/repositories/"
#remoteRepos=(dev_basics cache_io data_hub n3net colanet lidia mvit frame2frame stnls detectron2 nlnet vrt)
remoteRepos=(stnls dev_basics cache_io data_hub nlnet rvrt)
# localCodeDir="$HOME/Documents/packages/"
localCodeDir="$HOME/"

# if no output from the remote ssh cmd, bail out
if [ -z "$remoteRepos" ]; then
    echo "No results from remote repo listing (via SSH)"
    exit
fi

# for each repo found remotely, check if it exists locally
# assumption: name repo = repo.git, to be saved to repo (w/o .git)
# if dir exists, skip, if not, clone the remote git repo into it
for gitRepo in ${remoteRepos[@]}
do
  localRepoDir=$(echo ${localCodeDir}${gitRepo}|cut -d'.' -f1)
  # echo $localRepoDir
  if [ -d $localRepoDir ]; then
		echo -e "Directory $localRepoDir already exits, skipping ...\n"
	else
		cloneCmd="git clone https://github.com/gauenk/$gitRepo.git $localRepoDir"
		#cloneCmd="git clone git@github.com:$remoteUser/$gitRepo.git $localRepoDir"
		cloneCmdRun=$($cloneCmd 2>&1)
		# echo -e "Running: \n$ $cloneCmd"
		echo -e "${cloneCmd}"
	fi
done
