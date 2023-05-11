## Build and push to docker hub
```bash
docker build -t joshuapeddle128/imagetransfer-server:0.0.1 .

docker buildx build --platform linux/amd64,linux/arm64 -t joshuapeddle/imagetransfer-server:0.0.2 --push .