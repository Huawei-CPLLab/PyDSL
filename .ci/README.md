# pydsl/pydsl-ci Docker image maintenance instructions

The Autotest CI depends on the `pydsl-ci` docker image. The [dockerbuild.yml](../.github/workflows/dockerbuild.yml) provides a one-click build workflow to rebuild and update the docker image.

You would need to rebuild and update the docker image when any of the following happens:
1. The version of any of the following dependencies changes (current version number enclosed in parentheses):
   1. Python (3.12)
   2. LLVM/MLIR (19.1.7)
2. `requirements.txt` changes (i.e., we need additional Python packages for CI)

To do a rebuild, you only need to go to the actions tab, click on the workflow, and trigger it manually. If you want to build the image yourself, you can follow these steps:
1.  Make sure you have run `git submodule update --init --recursive` to obtain the latest submodule code (especially llvm-project).
2.  Run `docker build -f .ci/Dockerfile .` and wait for completion. Remember the hash of this docker image.

You can then test if this docker image works by (1) `docker run -it --rm <hash of docker image>` to start the container, (2) git checkout this repo, and (3) do whatever tests needed. The `--rm` flag removes this docker container after execution so no cleanup is needed.

In case you want to push the image manually, you need to:
1.  Tag the docker image with `ghcr.io/huawei-cpllab/pydsl-ci:latest`
2.  Create a personal PAT with permission to `write:package`, and save it to an environment variable (e.g., `export GHCR_PAT=<PAT>`)
3.  Run `echo $GHCR_PAT | docker login ghcr.io -u <your GitHub account> --password-stdin` to login ghcr. Remember to fill in your github account.
4.  Run `docker push ghcr.io/huawei-cpllab/pydsl-ci:latest` to push the image
