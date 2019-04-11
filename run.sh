#!/usr/bin/env bash
#
# © 2018 Konstantin Gredeskoul, All Rights Reserved.
# MIT License
#
# WARNING: This BASH script is completely optional. You don't need it to build this project.
#
# If you choose to run this script to build the project, run:
#
#     $ ./run.sh
#
# It will clean, build and run the tests.
#
#

( [[ -n ${ZSH_EVAL_CONTEXT} && ${ZSH_EVAL_CONTEXT} =~ :file$ ]] || \
  [[ -n $BASH_VERSION && $0 != "$BASH_SOURCE" ]]) && _s_=1 || _s_=0

export _s_
export ProjectRoot=$(pwd)
export BuildDir="${ProjectRoot}/build/run"
export BashLibRoot="${ProjectRoot}/bin/lib-bash"
export LibBashRepo="https://github.com/kigster/lib-bash"

# We are using an awesome BASH library `lib-bash` for prettifying the output, and
# running commands through their LibRun framework.
IntellGraph::lib-bash() {
  [[ ! -d ${BashLibRoot} ]] && curl -fsSL https://git.io/fxZSi | /usr/bin/env bash
  [[ ! -d ${BashLibRoot} ]] && { 
    printf "Unable to git clone lib-bash repo from ${LibBashRepo}"
    exit 1
  }
  
  if [[ -f ${BashLibRoot}/Loader.bash ]]; then
    cd ${BashLibRoot} > /dev/null
    git reset --hard origin/master 2>&1 | cat >/dev/null
    git pull 2>&1 | cat >/dev/null
    [[ -f Loader.bash ]] && source Loader.bash
    cd ${ProjectRoot}
  else
    printf "\nERROR: unable to find lib-bash library from ${LibBashRepo}!\n"
    exit 1
  fi

  run::set-all show-output-off abort-on-error
}

IntellGraph::header() {
  h1::purple "IntellGraph: A Deep Learning Framework in C++"
  local OIFC=${IFC}
  IFS="|" read -r -a gcc_info <<< "$(gcc --version 2>&1 | tr '\n' '|')"
  export IFC=${OIFC}
  h1 "${bldylw}GCC" "${gcc_info[1]}" "${gcc_info[2]}" "${gcc_info[3]}" "${gcc_info[4]}"
  h1 "${bldylw}GIT:    ${bldblu}$(git --version)"
  h1 "${bldylw}CMAKE:  ${bldblu}$(cmake --version | tr '\n' ' ')"
}

IntellGraph::setup() {
  hl::subtle "Creating Build Folder..."
  run "mkdir -p build/run"

  [[ -f .idea/workspace.xml ]] || cp .idea/workspace.xml.example .idea/workspace.xml
}

IntellGraph::clean() {
  hl::subtle "Cleaning output folders..."
  run 'rm -rf bin/d* include/d* lib/*'
}

IntellGraph::build() {
  run "cd build/run"
  run "cmake ../.. "
  run "make -j 12"
  run "make install | egrep -v 'gmock|gtest'"
  run "cd ${ProjectRoot}"
}

IntellGraph::tests() {
  if [[ -f bin/Intellgraph ]]; then
    run::set-next show-output-on
    run "echo && bin/Intellgraph"
  else
    printf "${bldred}Can't find installed executable ${bldylw}bin/Intellgraph.${clr}\n"
    exit 2
  fi
}

#IntellGraph::examples() {
#  [[ ! -f bin/IntellGraph ]] && {
#    error "You don't have the cmpiled binary yet".
#    exit 3
#  }
#
#  run::set-all show-output-on
#
#  hr
#  run "bin/IntellGraph 11 7"
#  hr
#  run "bin/IntellGraph 1298798375 94759897"
#  hr
#  run "bin/IntellGraph 78 17"
#  hr
#
#}

main() {
  IntellGraph::lib-bash
  IntellGraph::header
  IntellGraph::setup
  IntellGraph::build
  IntellGraph::tests
  IntellGraph::examples
}

(( $_s_ )) || main
