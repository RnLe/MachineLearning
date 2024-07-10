pytest:
	while true; do inotifywait -e modify,close_write,moved_to,move,create,delete $(shell find -maxdepth 2 -name '*.cu' -or -name '*.py'); pytest .; done
.PHONY: pytest
pytestv:
	while true; do inotifywait -e modify,close_write,moved_to,move,create,delete $(shell find -maxdepth 2 -name '*.cu' -or -name '*.py'); pytest . -vv; done
.PHONY: pytestv

