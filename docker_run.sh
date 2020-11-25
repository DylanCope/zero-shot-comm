sudo service docker start
sudo docker run --runtime nvidia -it --rm --name tf -v %(pwd):/tf/zero-shot-comm -p 8888:8888 -p 6060:6060 dc/zsc:latest
sudo service docker stop
