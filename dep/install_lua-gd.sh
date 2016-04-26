
#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################

sudo apt-get update
sudo apt-get install -qqy liblua5.1-0-dev
sudo apt-get install -qqy libgd-dev
sudo apt-get update


rm -rf lua-gd
echo "Installing Lua-GD ... "
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=\/home\/taey16\/torch\/install\/bin\/luajit" Makefile
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
rm -rf lua-gd
echo "Lua-GD installation completed"

