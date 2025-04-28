#/bin/bash
set -ex
pip3 install yapf --upgrade
yapf -ir -vv --style ./style.yapf .