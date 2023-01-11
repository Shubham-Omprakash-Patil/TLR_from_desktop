export DISPLAY="$(grep nameserver /etc/resolv.conf | sed 's/nameserver //'):0.0"
docker run -it --rm --name zeta -d \
-v /home/junho/py_ws:/home/junho/py_ws \
-v /mnt/c/database:/home/junho/database \
--gpus=all -e DISPLAY=$DISPLAY \
--network host --privileged \
--pid host \
--runtime=nvidia \
--shm-size=5.0gb \
--ipc=host \
junho:work
