
if [ ! -d "images" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/images.tar.gz
  tar zxf images.tar.gz
fi

MODEL_PATH="${PRETRAIN_MODEL_PATH}${MODEL}_oneflow"
IMAGE="cat.jpg" # change this line
IMAGE_NAME=${IMAGE%%.*}
STYLE_MODEL="sketch" # choose from sketch, candy, mosaic, rain_princess, udnie
CONTENT="images/content-images/${IMAGE}"
OUTPUT="${IMAGE_NAME}_${STYLE_MODEL}.jpg"

python3 neural_style.py \
    --content-image $CONTENT \
    --style-model $STYLE_MODEL \
    --output-image $OUTPUT
