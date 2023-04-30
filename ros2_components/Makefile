SRCS = ros_ws .
POETRY_PREFIX = poetry run
CONTAINER_NAME = yamamiya-pm-ros2
REGISTORY_NAME = nsakairi
DEFAULT_TAG = latest

format:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) black $$SRC --config pyproject.toml; done
	@for SRC in $(SRCS); do $(POETRY_PREFIX) isort $$SRC --profile black; done

build-app:
	docker-compose build --no-cache app

push-image:
	docker tag $(CONTAINER_NAME):$(DEFAULT_TAG) $(REGISTORY_NAME)/$(CONTAINER_NAME):$(DEFAULT_TAG)
	docker push $(REGISTORY_NAME)/$(CONTAINER_NAME):$(DEFAULT_TAG)
	docker rmi $(CONTAINER_NAME):$(DEFAULT_TAG)

build-app-and-push:
	docker-compose build --no-cache app
	docker push $(REGISTORY_NAME)/$(CONTAINER_NAME):$(DEFAULT_TAG)

build-run-app:
	docker-compose up -d --build app

exec-up:
	docker exec -it getter-robo 

clear_app_container:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

clear_images:
	docker stop $(docker ps -q)
	docker rm $(docker ps -q)
	docker rmi $(docker images -q)