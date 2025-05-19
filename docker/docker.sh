sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker

sudo docker build -t myapp .

docker tag myapp eakarsun4/myapp:latest

docker push eakarsun4/myapp:latest

sudo docker pull <your-dockerhub-username>/<your-image-name>:<tag>

sudo docker run -d -p 8000:8000 myapp

sudo docker run -it -p 8000:8000 myapp /bin/bash

docker tag myapp eakarsun4/myapp:latest
docker push eakarsun4/myapp:latest

sudo docker pull eakarsun4/myapp:latest

docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t eakarsun4/myapp:latest --push .


