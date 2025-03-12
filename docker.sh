#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="quiver-server"
CONTAINER_NAME="quiver-server"
PORT=8080
DATA_DIR="./data"
REBUILD=false
DETACH=true

# Print usage information
function print_usage {
    echo -e "${YELLOW}Usage:${NC} $0 [OPTIONS]"
    echo -e "${YELLOW}Options:${NC}"
    echo "  -h, --help                 Show this help message"
    echo "  -i, --image NAME           Set the Docker image name (default: quiver-server)"
    echo "  -c, --container NAME       Set the container name (default: quiver-server)"
    echo "  -p, --port PORT            Set the port to expose (default: 8080)"
    echo "  -d, --data-dir DIR         Set the data directory to mount (default: ./data)"
    echo "  -r, --rebuild              Force rebuild the Docker image"
    echo "  -f, --foreground           Run in foreground (not detached)"
    echo "  --stop                     Stop the running container"
    echo "  --remove                   Remove the container"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -c|--container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -r|--rebuild)
            REBUILD=true
            shift
            ;;
        -f|--foreground)
            DETACH=false
            shift
            ;;
        --stop)
            echo -e "${YELLOW}Stopping container ${CONTAINER_NAME}...${NC}"
            docker stop ${CONTAINER_NAME} 2>/dev/null || echo -e "${RED}Container ${CONTAINER_NAME} is not running.${NC}"
            exit 0
            ;;
        --remove)
            echo -e "${YELLOW}Removing container ${CONTAINER_NAME}...${NC}"
            docker rm -f ${CONTAINER_NAME} 2>/dev/null || echo -e "${RED}Container ${CONTAINER_NAME} does not exist.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Creating data directory: $DATA_DIR${NC}"
    mkdir -p "$DATA_DIR"
fi

# Check if the container is already running
if docker ps | grep -q ${CONTAINER_NAME}; then
    echo -e "${RED}Container ${CONTAINER_NAME} is already running.${NC}"
    echo -e "Use '${YELLOW}$0 --stop${NC}' to stop it or '${YELLOW}$0 --remove${NC}' to remove it."
    exit 1
fi

# Check if we need to rebuild or if the image doesn't exist
if [ "$REBUILD" = true ] || ! docker images | grep -q ${IMAGE_NAME}; then
    echo -e "${GREEN}Building Docker image: ${IMAGE_NAME}...${NC}"
    
    # Build for arm64 architecture explicitly
    echo -e "${YELLOW}Building for platform: linux/arm64${NC}"
    docker build --platform=linux/arm64 -t ${IMAGE_NAME} .
else
    echo -e "${GREEN}Using existing Docker image: ${IMAGE_NAME}${NC}"
fi

# Run the container
echo -e "${GREEN}Starting container: ${CONTAINER_NAME}...${NC}"
if [ "$DETACH" = true ]; then
    docker run -d \
        --platform=linux/arm64 \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8080 \
        -v ${DATA_DIR}:/app/data \
        -e QUIVER_LOG_LEVEL=info \
        ${IMAGE_NAME}
    
    echo -e "${GREEN}Container started in detached mode.${NC}"
    echo -e "Access the API at http://localhost:${PORT}"
    echo -e "View logs with: ${YELLOW}docker logs ${CONTAINER_NAME}${NC}"
    echo -e "Stop with: ${YELLOW}$0 --stop${NC}"
else
    echo -e "${GREEN}Container starting in foreground mode. Press Ctrl+C to stop.${NC}"
    docker run --rm \
        --platform=linux/arm64 \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8080 \
        -v ${DATA_DIR}:/app/data \
        -e QUIVER_LOG_LEVEL=info \
        ${IMAGE_NAME}
fi 