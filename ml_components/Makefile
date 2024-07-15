SRCS = test ml_components . 
#POETRY_PREFIX = poetry run
POETRY_PREFIX = rye run
CONTAINER_NAME = yamamiya-pm-ml
REGISTORY_NAME = nsakairi
DEFAULT_TAG = latest

format:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) black $$SRC --config pyproject.toml; done
	@for SRC in $(SRCS); do $(POETRY_PREFIX) isort $$SRC --profile black; done

build-app:
	docker-compose build --no-cache yamamiya-pm-ml
push-image:
	docker tag $(CONTAINER_NAME):$(DEFAULT_TAG) $(REGISTORY_NAME)/$(CONTAINER_NAME):$(DEFAULT_TAG)
	docker push $(REGISTORY_NAME)/$(CONTAINER_NAME):$(DEFAULT_TAG)
# sha256:d805e62840d43d7617960ef70d22b6ef2a9dbb714d744e850da9563d2c8aed31

build-app-and-push:
	docker-compose build --no-cache 
	docker push $(REGISTORY_NAME)/$(CONTAINER_NAME):$(DEFAULT_TAG)

build-run-app:
	docker-compose up -d --build yamamiya-pm-ml

exec-up:
	docker exec -it getter-robo 

clear_app_container:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

clear_images:
	docker stop $(docker ps -q)
	docker rm $(docker ps -q)
	docker rmi $(docker images -q)