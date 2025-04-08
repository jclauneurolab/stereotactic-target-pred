#!/bin/bash
# Build script for the PyCSC (Python Christoffel Symbols Calculator) grpahical user interface
echo "Building Stereotaxy Prediction GUI..."
#
# Set some parameters
#
VERSION=$(date +%y.%m.%d.%H%M)
# (following line from <https://stackoverflow.com/a/246128>)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# (following line from <https://stackoverflow.com/a/8426110>)
REPO_DIR="$(dirname "$SCRIPT_DIR")"
#
# Load custom parameters
#
source ${SCRIPT_DIR}/Docker_env

RUN=${1:-false}
#
# Build the project
#
cd ${REPO_DIR}  # necessary so Docker can access other folders within the repo
docker build -t stereotaxy_gui:${VERSION} \
        --platform=linux/amd64 -f docker/Dockerfile .
#
echo "Finishing building stereotaxy_gui:${VERSION}"
#
# Run the project
#
if [[ ${RUN} = true ]]; then
    echo "Now running stereotaxy_gui_v${VERSION}..."
    # https://docs.docker.com/config/containers/container-networking/#published-ports
    docker run --interactive \
            --tty \
            --publish 5001:5000 \
            --name stereotaxy__gui_v${VERSION} \
            -d stereotaxy_gui:${VERSION}
    #
    docker logs stereotaxy_gui_v${VERSION}
    #
    echo "DONE! Access the stereotaxy_gui_v${VERSION} instance via localhost port 5000."
else
    echo "DONE! The stereotaxy_gui_v${VERSION} has been built."
fi