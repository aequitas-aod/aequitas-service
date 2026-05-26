.PHONY: install add add-dev add-git remove update lock shell

install:
	poetry install

## Add a PyPI package:       make add pkg=requests
## With version constraint:  make add pkg="requests>=2.28"
add:
	poetry add $(pkg)

## Add a dev-only package:   make add-dev pkg=pytest
add-dev:
	poetry add --group dev $(pkg)

## Add a Git-hosted package: make add-git url=https://github.com/org/repo
add-git:
	poetry add git+$(url)

## Remove a package:         make remove pkg=requests
remove:
	poetry remove $(pkg)

## Update all packages to their latest allowed versions
update:
	poetry update

## Regenerate poetry.lock without upgrading packages
lock:
	poetry lock --no-update

## Activate the virtual environment
shell:
	poetry shell
