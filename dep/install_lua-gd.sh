
#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################

# install dep. for lua-gd
sudo apt-get update
sudo apt-get install -qqy liblua5.1-0-dev
sudo apt-get install -qqy libgd-dev
sudo apt-get update

#shell$ git clone https://github.com/ittner/lua-gd.git
#shell$ cd lua-gd
#shell$ luarocks make
#Warning: variable CFLAGS was not passed in build_variables
#gcc -o gd.lo -c `gdlib-config --features |sed -e "s/GD_/-DGD_/g"` -O3 -Wall -fPIC -fomit-frame-pointer `gdlib-config --cflags` `pkg-config lua5.1 --cflags` -DVERSION=\"2.0.33r3\" luagd.c
#gcc -o gd.so gd.lo -shared `gdlib-config --ldflags` `gdlib-config --libs` -lgd
#lua5.1 test_features.lua
#make: lua5.1: Command not found
#make: *** [test] Error 127
#shell$ cp gd.so ~/torch/install/lib/


rm -rf lua-gd
echo "Installing Lua-GD ... "
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=\/home\/taey16\/torch\/install\/bin\/luajit" Makefile
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
rm -rf lua-gd
echo "Lua-GD installation completed"

