#!/usr/bin/env bash

echo "Installing Lua-GD ... "

HOME=/home/taey16/
cd $HOME
rm -rf lua-gd
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Lua-GD installation completed"

