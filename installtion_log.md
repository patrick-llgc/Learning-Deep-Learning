# installtion log

- Install Pycharm
- Use `nvidia-smi` in linux terminal to show how many GPUs are installed

### Change linux on windows terminal color
- Add the following to `~/.bashrc` file.
```bash
LS_COLORS='rs=0:di=1;35:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arj=01;31:*.taz=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lz=01;31:*.xz=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.axv=01;35:*.anx=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.axa=00;36:*.oga=00;36:*.spx=00;36:*.xspf=00;36:'
  
export LS_COLORS

# this will change your prompt format
# created from http://bashrcgenerator.com/
PS1="[ \[$(tput sgr0)\]\[\033[38;5;10m\]\u\[$(tput sgr0)\]\[\033[38;5;15m\] @ \[$(tput sgr0)\]\[\033[38;5;9m\]\h\[$(tput sgr0)\]\[\033[38;5;15m\] \w ]\\$ \[$(tput sgr0)\]"
```
### Invoke iPython
```bash
alias ipy="python -c 'import IPython; IPython.terminal.ipapp.launch_new_instance()'"
```

### Change tmux config

Change the key binding from ctrl + b to ctrl + a
```
# remap prefix to Control + a
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# force a reload of the config file
unbind r
bind r source-file ~/.tmux.conf

# quick pane cycling
unbind ^A
bind C-a select-pane -t :.+


set-window-option -g aggressive-resize
```

### Change vim color scheme

- `echo "set background=dark" >> .vimrc`
- Install `XMing`

### Git squash
Create a git alias called git squash by type in the following.

```bash
git config --global alias.squash '!f(){ git reset --soft HEAD~${1} && git commit --edit -m"$(git log --format=%B --reverse HEAD..HEAD@{1})"; };f'
```
Now use the following command to combine the last two commits. Do not push before you squash!
```git squash 8```

### Useful git hooks

#### remove output cells from ipynb

```bash

#!/bin/sh
#
# strip output of IPython Notebooks
# add this as `.git/hooks/pre-commit`
# to run every time you commit a notebook
#
# requires `nbstripout` to be available on your PATH
#

if git rev-parse --verify HEAD >/dev/null 2>&1; then
   against=HEAD
else
   # Initial commit: diff against an empty tree object
   against=4b825dc642cb6eb9a060e54bf8d69288fbee4904
fi
 
# Find notebooks to be committed
(
IFS='
'
# Note by Patrick Liu: the -z option sometimes combines all lines into one
# NBS=`git diff-index -z --cached $against --name-only | grep -a '.ipynb' | uniq`
NBS=`git diff-index --cached $against --name-only | grep -a '.ipynb' | uniq`

for NB in $NBS ; do
    echo "Removing outputs from $NB"
    python /media/Users/pliu/nbstripout/nbstripout.py "$NB"
    git add "$NB"
done
)

exec git diff-index --check --cached $against --
```

#### Limit file size with pre-commit hook

```bash
#!/bin/sh
hard_limit=$(git config hooks.filesizehardlimit)
soft_limit=$(git config hooks.filesizesoftlimit)
: ${hard_limit:=10000000}
: ${soft_limit:=500000}
list_new_or_modified_files()
{
    git diff --staged --name-status|sed -e '/^D/ d; /^D/! s/.\s\+//'
}
unmunge()
{
    local result="${1#\"}"
    result="${result%\"}"
    env echo -e "$result"
}
check_file_size()
{
    n=0
    while read -r munged_filename
    do
        f="$(unmunge "$munged_filename")"
        h=$(git ls-files -s "$f"|cut -d' ' -f 2)
        s=$(git cat-file -s "$h")
        if [ "$s" -gt $hard_limit ]
        then
            env echo -E 1>&2 "ERROR: hard size limit ($hard_limit) exceeded: $munged_filename ($s)"
            n=$((n+1))
        elif [ "$s" -gt $soft_limit ]
        then
            env echo -E 1>&2 "WARNING: soft size limit ($soft_limit) exceeded: $munged_filename ($s)"
        fi
    done
    [ $n -eq 0 ]
}
list_new_or_modified_files | check_file_size

```



### Install various tools

- ```
  # https://joshpeng.github.io/post/wsl/
  sudo apt-get update
  sudo apt-get upgrade
  
  apt-get install firefox
  sudo apt-get install zsh git libqtgui4 xserver-xorg-video-dummy
  echo "export DISPLAY=localhost:0.0" >> ~/.bashrc
  
  # install sublime
  sudo add-apt-repository ppa:webupd8team/sublime-text-3
  sudo apt-get update
  sudo apt-get install sublime-text-installer
  subl
  
  # install sublime
  wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
  echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
  sudo apt-get update
  sudo apt-get install sublime-text
  
  # get pip for python3
  sudo apt-get install python3-pip
  
  # install virtualenv 
  sudo pip3 install virtualenv
  # install virtualenvwrapper, a higher level virtualenv tool
  # http://docs.python-guide.org/en/latest/dev/virtualenvs/
  # NB that `pip3` equals to `python3 -m pip`
  # you may need to `pip install --upgrade pip`
  # you may need to `sudo pip3 install -U setuptools`
  sudo pip3 install virtualenvwrapper
  export WORKON_HOME=~/Envs
  export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
  source /usr/local/bin/virtualenvwrapper.sh
  
  # make virtualenv (e.g., py3) and workon it
  mkvirtualenv --python=/usr/bin/python3 py3
  workon py3
  
  # install tensorflow
  # Do not use `pip3 install --upgrade tensorflow-gpu`
  # Note that tf 1.3 requires libcudnn.so.6
  pip3 install tensorflow-gpu==1.1
  
  # NB. sometimes it may be necessary to upgrade to 1.2 to run tensorboard properly
  # on DGX, cudnn7 is installed, so you have to use tensorflow 1.3 or above.
  # or specify $LD_LIBRARY_PATH and copy cudnn5 to the folder it points to 
  
  # install tkinter to run matplotlib properly
  sudo apt-get install python3-tk
  
  ```

### In WSL

- Mount network drive in WSL

  ```bash
  # This is not available until windows 10 build 16299
  # sudo mkdir /media/Users
  ```

- Download and install pycharm 

  `https://confluence.jetbrains.com/display/PYH/Installing+PyCharm+on+Linux+according+to+FHS`​

### Conda vs pip

If you like to use anaconda vs pip, then do the following.
```bash
  # install anaconda
  wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
  bash Anaconda3-4.2.0-Linux-x86_64.sh
  conda update --all --y
  
  # update conda to fix multi-user install 
  conda update conda
  conda create -n py27 python=2.7 anaconda
  # use `conda env list` to check which env is it under
```

### Multiple versions of CUDA on the same machine

We can install multiple version of CUDA on the same machine, and link it against different vrittualenv. Here I document the steps to install CUDA 8 and CUDA 9 (required for TF 1.5+) on the same machine.

1. Suppose we have cuda 8.0 installed. 
  ```bash
  $ nvcc --version
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2016 NVIDIA Corporation
  Built on Tue_Jan_10_13:22:03_CST_2017
  Cuda compilation tools, release 8.0, V8.0.61
  ```

2. Note that CUDA 9 requires Ubuntu16.04 LTS. Please update OS to 16.04 LTS before proceeding

3. Follow the instructions on 
    https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77
    We can achieve the same with virtualenvwrapper, instead of Anaconda
    https://stackoverflow.com/a/11134336/5304522



### Git commit message guideline

[This](https://chris.beams.io/posts/git-commit/) is a good blog about how to write good commit messages.

- Separate subject from body with a blank line
- Limit the subject line to 50 characters
- Capitalize the subject line
- Do not end the subject line with a period
- Use the imperative mood in the subject line
- Wrap the body at 72 characters
- Use the body to explain what and why vs. how


### Recommended git workflow

Please follow the git workflow as follows

1) git 
```bash
$ cd /your/directory
$ git clone your_repo_url
```

2) Create a new branch. It will contain the latest files of your master branch repository
	```$ git branch new_branch```

3) Change git branch to the new_branch
	```$ git checkout new_branch```

4) Make changes and commit.

   ```bash
$ git diff theFileYouChanged (Make sure the changes are what you want)
$ git add theFileYouChanged (Only add the files you know you have changed)
$ git commit -m “PROJ-001 #Initial commit” (for research team, no need to prepend commit messages with JIRA numbers. Start each commit message with a verb. See Git commit standard for details.)
$ git push
   ```

5) When job is finished on this branch, merge latest changes from “master” branch into this branch and issue pull request.

   ```bash
$ git checkout master (goes to master branch)
$ git fetch
$ git diff master origin/master (know what has changed on master)
$ git pull
$ git checkout new_branch
$ git merge master (Note that there may be a merge conflict. In that case, fix the conflict in conflicted files, then git committhen git commit)
$ git push
   ```

6) Go to github.com and issue a pull request. Remember to assign a reviewer that knows about your project.