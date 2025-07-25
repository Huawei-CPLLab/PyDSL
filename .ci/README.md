# pydsl/pydsl-ci Docker image maintenance instructions

The CI of this repository depends on the docker image `pydsl/pydsl-ci:latest`. Files involved in building this image:
* `Dockerfile` for constructing the docker image
* `docker_fileprep.sh` for copying files outside this `.ci` directory into this building context

## Update docker image

You would need to rebuild and update the docker image when any of the following happens:
1. The version of any of the following dependencies changes (current version number enclosed in parentheses):
   1. Python (3.12)
   2. LLVM/MLIR (19.1.7)
2. `requirements.txt` changes (i.e., we need additional Python packages for CI)

Steps:
1.  Make sure you have run `git submodule update --init --recursive` to obtain the latest submodule code (especially llvm-project).
2.  `cd` into the `.ci` directory.
3.  Run `docker_fileprep.sh`. When executing the docker build, the script cannot access anything outside. This command will copy files into the `.ci` directory to make them available for docker build.
4.  Run `docker build .` and wait for completion. Remember the hash of this docker image.
5.  Test if this docker image works by (1) `docker run -it --rm <hash of docker image>` to start the container, (2) git checkout this repo, and (3) do whatever tests needed. The `--rm` flag removes this docker container after execution so no cleanup is needed.
6.  If the docker image works, run `docker image tag <hash of docker image> pydsl/pydsl-ci:latest` to tag this image.
7.  Use `docker login` to log in to the `pydsl` account and `docker push pydsl/pydsl-ci:latest` to push the image.
